"""Select ONE W with the existing SVD/codebook selector, then retain all Z.

The reference RIS pattern and pilot channel are fixed before variance MC.
This is the repository's precoder-selection rule, not q05/error optimization.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import torch

from .channel_primitives import generate_cascaded_ch
from .channel_realizations import generate_native_link_chunk
from .codebook import generate_codebook_rank1, select_precoder_rank1
from .validation import row_to_link_configs


@torch.inference_mode()
def select_one_precoder(prepared, *, reference_z_index=4, pilot_seed_offset=0):
    """Return (prepared_with_one_W, provenance, pilot_arrays).

    Keep the original 512-pattern pool built by prepare_teacher_bank unchanged.
    Generate ONE independent BR/RU pilot pair, using gamma[reference_z_index].
    SVD of F_pilot -> V -> existing select_precoder_rank1(full_codebook,V).
    Hold this W fixed for every Z and every subsequent evaluation realization.

    reference_z_index defaults to 4: the first global-random candidate after
    four anchors in the current teacher pool. This is an explicit reference
    phase choice, not a search over Z. The selected codeword can be outside
    the evenly-spaced original 32-W pool. wCandidate=0 in the returned object
    is only a local index; WIdx identifies the actual codeword.
    """
    if isinstance(reference_z_index,bool) or int(reference_z_index)!=reference_z_index:
        raise ValueError('reference_z_index must be an integer')
    if not 0<=reference_z_index<len(prepared.gamma):
        raise ValueError('reference_z_index outside the fixed Z pool')
    if isinstance(pilot_seed_offset,bool) or int(pilot_seed_offset)!=pilot_seed_offset or pilot_seed_offset<0:
        raise ValueError('pilot_seed_offset must be a nonnegative integer')
    reference_z_index=int(reference_z_index)
    env=prepared.static_env;dev=env.muBR.device
    parity=env.muBR.dtype==torch.complex128
    if env.muBR.dtype not in (torch.complex64,torch.complex128):
        raise ValueError('Expected complex environment moments')
    row=prepared.row_physical
    seed0=int(row.ch_seed)
    seed_br=seed0+10_000_019+int(pilot_seed_offset)
    seed_ru=seed0+20_000_033+int(pilot_seed_offset)
    evaluation_seeds={seed0+70_000_121,seed0+80_000_147}
    if seed_br==seed_ru or {seed_br,seed_ru}&evaluation_seeds:
        raise ValueError('Pilot streams must differ from the default evaluation streams')
    if not 0<=min(seed_br,seed_ru) or max(seed_br,seed_ru)>=2**63:
        raise ValueError('Pilot seed out of range')
    br,ru=row_to_link_configs(row)
    gen_br=torch.Generator(device=dev).manual_seed(seed_br)
    gen_ru=torch.Generator(device=dev).manual_seed(seed_ru)
    hbr=generate_native_link_chunk(br,1,generator=gen_br,device=dev,parity=parity)
    hru=generate_native_link_chunk(ru,1,generator=gen_ru,device=dev,parity=parity)
    gamma_ref=prepared.gamma[reference_z_index]
    F=generate_cascaded_ch(hbr,hru,gamma_ref)[0]
    if not bool(torch.isfinite(F).all()) or float(torch.linalg.vector_norm(F))==0:
        raise FloatingPointError('Pilot cascaded channel must be finite and nonzero')
    _,s,Vh=torch.linalg.svd(F,full_matrices=False)
    cb=generate_codebook_rank1(2,int(row.nT1),int(row.nT2),1,1,device=dev,parity=parity)
    w,idx=select_precoder_rank1(cb,Vh.conj().T)
    matches=torch.nonzero((prepared.WIdx==idx[None,:]).all(dim=1),as_tuple=False).reshape(-1)
    original_index=int(matches[0]) if len(matches) else -1
    power=float(torch.sum(torch.abs(F@w)**2))
    old_powers=torch.sum(torch.abs(F@prepared.W.T)**2,dim=0)
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance={
        'selectionMethod':'SVD_then_existing_select_precoder_rank1',
        'selectionReferenceZ':reference_z_index,
        'selectionReferenceType':str(prepared.z_candidate_type[reference_z_index]),
        'selectionPilotSamples':1,
        'selectionSeedBR':seed_br,'selectionSeedRU':seed_ru,
        'selectionPilotSeedOffset':int(pilot_seed_offset),
        'selectedOriginalWCandidate':original_index,
        'selectionI11':int(idx[0]),'selectionI12':int(idx[1]),'selectionI2':int(idx[2]),
        'selectionPilotPower':power,
        'selectionPilotSVDUpperBound':float(s[0]**2),
        'selectionOriginalPoolMaxPilotPower':float(old_powers.max()),
        'selectionCodeSHA256':code_hash,
    }
    physical=prepared.row_physical.copy()
    for key,value in provenance.items():
        physical[key]=value
    selected=replace(prepared,row_physical=physical,W=w[None,:],WIdx=idx[None,:])
    # Z/gamma objects are deliberately preserved rather than regenerated for W.
    arrays={'F_pilot':F.detach().cpu().numpy(),
            'gamma_reference':gamma_ref.detach().cpu().numpy(),
            'z_reference':prepared.z[reference_z_index].detach().cpu().numpy(),
            'singular_values':s.detach().cpu().numpy(),
            'original_pool_W':prepared.W.detach().cpu().numpy(),
            'original_pool_WIdx':prepared.WIdx.detach().cpu().numpy()}
    return selected,provenance,arrays


def compare_with_full_pool(selected_result, baseline_folder, selected_gamma, *, selected_row):
    """Descriptive same-bank/Z comparison with an already completed 32-W run.

    This does not infer that selector W globally minimizes variance error.
    It checks the saved Z/gamma identity before comparing candidate errors.
    Baseline and selected evaluation streams can differ; report that separately
    from the strict within-baseline selected-W comparison, when WIdx exists.
    """
    import pandas as pd
    import json
    from .variance_comparison import summarize_comparisons
    baseline_folder=Path(baseline_folder)
    manifest_path=baseline_folder/'manifest.json'
    candidates_path=baseline_folder/'candidates.npz'
    table_path=baseline_folder/'comparison.parquet'
    if not all(p.exists() for p in (manifest_path,candidates_path,table_path)):
        return {'baselineAvailable':False}
    original_row=json.loads(manifest_path.read_text())['row']
    keys=('fc','ris_x','ris_y','ris_z','gnb_x','gnb_y','gnb_z','ue_x','ue_y','ue_z',
          'nT1','nT2','nR1','nR2','nRIS1','nRIS2','scenario_BR','scenario_RU')
    for key in keys:
        a,b=original_row.get(key),selected_row.get(key)
        equal=a==b if isinstance(a,str) or a is None or b is None else bool(np.isclose(a,b,rtol=1e-9,atol=1e-10))
        if a is None or b is None or not equal:
            return {'baselineAvailable':True,'baselineComparable':False,'reason':f'Different physical row: {key}'}
    if torch.is_tensor(selected_gamma):
        selected_gamma=selected_gamma.detach().cpu().numpy()
    with np.load(candidates_path,allow_pickle=False) as data:
        baseline_gamma=data['gamma']
        z_ids=data['zCandidate']
    selected_ids=selected_result.zCandidate.to_numpy(dtype=np.int64)
    lookup={int(z):i for i,z in enumerate(z_ids)}
    if any(int(z) not in lookup for z in selected_ids):
        return {'baselineAvailable':True,'baselineComparable':False,'reason':'Z index absent'}
    existing=baseline_gamma[[lookup[int(z)] for z in selected_ids]]
    if existing.shape!=selected_gamma.shape or not np.allclose(existing,selected_gamma,rtol=1e-10,atol=1e-12):
        return {'baselineAvailable':True,'baselineComparable':False,'reason':'Different gamma/Z pool or precision'}
    baseline=pd.read_parquet(table_path)
    baseline=baseline[baseline.zCandidate.isin(selected_ids)]
    if baseline.bankID.nunique()!=1 or int(baseline.bankID.iloc[0])!=int(selected_result.bankID.iloc[0]):
        return {'baselineAvailable':True,'baselineComparable':False,'reason':'Different bankID'}
    old=summarize_comparisons(baseline)
    new=summarize_comparisons(selected_result)
    out={'baselineAvailable':True,'baselineComparable':True,
         'fullPoolHopMedianAbsErrorPct':old['hopMedianAbsErrorPct'],
         'selectedHopMedianAbsErrorPct':new['hopMedianAbsErrorPct'],
         'fullPoolHopP95':float(baseline.loc[baseline.analyticValid,'hopAbsErrorPct'].quantile(.95)),
         'selectedHopP95':float(selected_result.loc[selected_result.analyticValid,'hopAbsErrorPct'].quantile(.95)),
         'fullPoolNMC':int(baseline.N_MC.iloc[0]),'selectedNMC':int(selected_result.N_MC.iloc[0])}
    match=np.ones(len(baseline),dtype=bool)
    for name in ('i11','i12','i2'):
        match &= baseline[name].to_numpy()==int(selected_result[name].iloc[0])
    old_selected=baseline.loc[match]
    out['selectorWPresentInOldPool']=bool(len(old_selected))
    if len(old_selected):
        out['sameOldStreamSelectedHopMedianAbsErrorPct']=float(old_selected.loc[old_selected.analyticValid,'hopAbsErrorPct'].median())
        out['sameOldStreamSelectedHopP95']=float(old_selected.loc[old_selected.analyticValid,'hopAbsErrorPct'].quantile(.95))
    return out
