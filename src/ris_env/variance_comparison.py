"""MC, existing Wick, and conditional-hop analytic variance on identical W/Z.

Uses the production 4000-bank preparation and stateful native MC engine.
This experiment has no dependency on MATLAB files or learned models.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .hop_power_variance import (
    METHOD_VERSION, power_variance_batch, prepare_br_moments,
    prepare_ru_covariance_blocks,
)
from .snr_statistics import prepare_w_state, evaluate_gamma_batch
from .stateful_mc import create_stateful_mc, advance_stateful_mc, snapshot_stateful_mc


def _numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _indices(values, size, name):
    if values is None:
        return np.arange(size,dtype=np.int64)
    a = np.asarray(list(values))
    if a.ndim != 1 or not len(a) or not np.issubdtype(a.dtype,np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if (a<0).any() or (a>=size).any() or len(np.unique(a))!=len(a):
        raise ValueError(f"Invalid or duplicate {name}")
    return a.astype(np.int64)


def _atomic_npz(path, **data):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**data)
    os.replace(tmp,path)


def _atomic_table(df,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    if path.suffix=='.parquet':
        df.to_parquet(tmp,index=False)
    else:
        df.to_csv(tmp,index=False)
    os.replace(tmp,path)


def _source_hash():
    # All code affecting candidate selection, moments, or RNG is tracked.
    names=('variance_comparison','hop_power_variance','teacher_pipeline',
           'environment','geometry_lsp','antenna','spatial_correlation',
           'snr_statistics','codebook','ris_response','label_engine',
           'validation','stateful_mc','channel_primitives','channel_realizations')
    root=Path(__file__).parent
    return {n:hashlib.sha256((root/(n+'.py')).read_bytes()).hexdigest() for n in names}


def _manifest(prepared,W,gamma,wi,zi,options):
    h=hashlib.sha256()
    for a in (W,gamma):
        a=np.ascontiguousarray(_numpy(a))
        h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    row=json.loads(prepared.row_physical.to_json())
    return dict(method=METHOD_VERSION,row=row,candidates_sha256=h.hexdigest(),
                w_indices=wi.tolist(),z_indices=zi.tolist(),options=options,
                source_hashes=_source_hash(),torch_version=str(torch.__version__))


def _guard_manifest(folder,manifest):
    path=folder/'manifest.json'
    encoded=json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise ValueError('Inputs, precision, code, or settings changed; use a new output directory.')
    else:
        if any(folder.iterdir()):
            raise ValueError('Nonempty output directory without manifest; choose another directory.')
        tmp=folder/'manifest.json.tmp';tmp.write_text(encoded);os.replace(tmp,path)


def _load_mc(path,state):
    if not path.exists():
        return
    with np.load(path,allow_pickle=False) as data:
        n=int(data['n_samples'])
        if n<0 or data['sum1'].shape!=tuple(state.sum1.shape) or data['sum2'].shape!=tuple(state.sum2.shape):
            raise ValueError('Invalid MC checkpoint')
        if not np.isfinite(data['sum1']).all() or not np.isfinite(data['sum2']).all():
            raise ValueError('Non-finite MC checkpoint')
        state.sum1.copy_(torch.as_tensor(data['sum1'],device=state.device))
        state.sum2.copy_(torch.as_tensor(data['sum2'],device=state.device))
        state.gen_br.set_state(torch.from_numpy(data['rng_br'].copy()))
        state.gen_ru.set_state(torch.from_numpy(data['rng_ru'].copy()))
        state.n_samples=n


def _save_mc(path,state):
    _atomic_npz(path,n_samples=state.n_samples,sum1=_numpy(state.sum1),sum2=_numpy(state.sum2),
                rng_br=_numpy(state.gen_br.get_state()),rng_ru=_numpy(state.gen_ru.get_state()))


def _deviations(reference,estimate):
    den=np.where(reference>0,reference,np.nan)
    ratio=estimate/den
    with np.errstate(divide='ignore',invalid='ignore'):
        log=np.where((reference>0)&(estimate>0),np.log(ratio),np.nan)
    return 100*(ratio-1),log


@torch.inference_mode()
def compare_prepared_bank(
    prepared, output_dir, *, n_mc=64_000, mc_chunk=256, w_chunk=4,
    z_chunk=64, analytic_z_chunk=4, checkpoint_samples=4096,
    w_indices=None, z_indices=None, seed_br=None, seed_ru=None, progress=print,
):
    """Run/resume a bank; return one row per selected W/Z.

    By default evaluates the entire existing 32 x 512 teacher candidate pool.
    MC comes from advance_stateful_mc; Wick from evaluate_gamma_batch. New
    analytic inputs never include MC moments. Precision is inherited from
    prepared.static_env; prepare_teacher_bank(parity=True) is recommended.
    Checkpoints preserve RNG states, so resuming does not repeat samples.
    Keep one writer per output directory. Complete banks return immediately
    after validating the manifest. No training/teacher labels are modified.
    """
    for name,value in dict(n_mc=n_mc,mc_chunk=mc_chunk,w_chunk=w_chunk,z_chunk=z_chunk,
                           analytic_z_chunk=analytic_z_chunk,checkpoint_samples=checkpoint_samples).items():
        if isinstance(value,bool) or int(value)!=value or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if n_mc<2 or checkpoint_samples%mc_chunk:
        raise ValueError('n_mc >= 2 and checkpoint_samples must be a multiple of mc_chunk')
    env=prepared.static_env;dev=env.muBR.device
    if env.muBR.dtype not in (torch.complex64,torch.complex128):
        raise ValueError('Expected complex channel moments')
    parity=env.muBR.dtype==torch.complex128
    wi=_indices(w_indices,len(prepared.W),'w_indices')
    zi=_indices(z_indices,len(prepared.gamma),'z_indices')
    W=prepared.W[torch.as_tensor(wi,device=dev)]
    gamma=prepared.gamma[torch.as_tensor(zi,device=dev)]
    options=dict(n_mc=int(n_mc),mc_chunk=int(mc_chunk),w_chunk=int(w_chunk),z_chunk=int(z_chunk),
                 analytic_z_chunk=int(analytic_z_chunk),checkpoint_samples=int(checkpoint_samples),
                 seed_br=None if seed_br is None else int(seed_br),
                 seed_ru=None if seed_ru is None else int(seed_ru),parity=parity,device_type=dev.type,
                 device_name=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'cpu')
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    manifest=_manifest(prepared,W,gamma,wi,zi,options)
    _guard_manifest(folder,manifest)
    result_path=folder/'comparison.parquet'
    if result_path.exists():
        return pd.read_parquet(result_path)
    candidate_path=folder/'candidates.npz'
    if not candidate_path.exists():
        _atomic_npz(candidate_path,W=_numpy(W),gamma=_numpy(gamma),
                    WIdx=_numpy(prepared.WIdx)[wi],z=_numpy(prepared.z)[zi],
                    wCandidate=wi,zCandidate=zi)
    # Existing MC implementation and seed convention, unchanged.
    state=create_stateful_mc(prepared.row_physical,W,gamma,mc_chunk=mc_chunk,
                            w_chunk=w_chunk,z_chunk=z_chunk,device=dev,parity=parity,
                            seed_br=seed_br,seed_ru=seed_ru)
    checkpoint=folder/'mc_checkpoint.npz'
    _load_mc(checkpoint,state)
    if state.n_samples>n_mc or (state.n_samples<n_mc and state.n_samples%mc_chunk):
        raise ValueError('MC checkpoint sample count is incompatible')
    t0=time.perf_counter()
    while state.n_samples<n_mc:
        advance_stateful_mc(state,min(checkpoint_samples,n_mc-state.n_samples))
        _save_mc(checkpoint,state)
        snap=snapshot_stateful_mc(state)
        _atomic_npz(folder/f'mc_at_{state.n_samples:07d}.npz',
                    meanEmp=_numpy(snap['meanEmp']),varEmp=_numpy(snap['varEmp']))
        if progress:
            progress(f"bank {int(prepared.row_raw.bankID)} MC {state.n_samples:,}/{n_mc:,} | {time.perf_counter()-t0:.1f}s")
    mc=snapshot_stateful_mc(state)
    mean_mc,var_mc=_numpy(mc['meanEmp']),_numpy(mc['varEmp'])
    # Report cancellation/clipping diagnostic from the unchanged MC accumulator.
    raw_mc=_numpy(state.sum2/n_mc-(state.sum1/n_mc)**2)
    del state
    # Shared full receiver covariance blocks, prepared once per bank.
    cg=prepare_ru_covariance_blocks(env)
    records=[]
    eps=np.finfo(np.float64 if parity else np.float32).eps
    tol=1e-8 if parity else 3e-4
    for k,w in enumerate(W):
        analytic_path=folder/f'analytic_w_{int(wi[k]):03d}.npz'
        if analytic_path.exists():
            with np.load(analytic_path,allow_pickle=False) as data:
                values={key:data[key] for key in data.files}
            if values['sigma2Hop'].shape!=(len(zi),):
                raise ValueError('Invalid analytic checkpoint shape')
        else:
            # Old Wick is independently computed through its production API.
            ws=prepare_w_state(env,w)
            old=evaluate_gamma_batch(ws,gamma,candidate_chunk=z_chunk)
            values={'muSNR':_numpy(old['muSNR']),'sigma2Wick':_numpy(old['sigma2Wick'])}
            del old,ws
            mx,cx=prepare_br_moments(env,w)
            parts={key:[] for key in ('muHop','sigma2Hop','wickFromHop',
                'varianceImagResidual','meanImagResidual','varConditionalMean',
                'expectedTraceV2','expectedMeanVMean')}
            for start in range(0,len(zi),analytic_z_chunk):
                stop=min(start+analytic_z_chunk,len(zi))
                mean,var,dg=power_variance_batch(mx,cx,env.muRU,cg,gamma[start:stop])
                parts['muHop'].append(_numpy(mean));parts['sigma2Hop'].append(_numpy(var))
                parts['wickFromHop'].append(_numpy(dg['varWickEffectiveGaussian']))
                for key in parts.keys()-{'muHop','sigma2Hop','wickFromHop'}:
                    parts[key].append(_numpy(dg[key]))
            values.update({key:np.concatenate(value) for key,value in parts.items()})
            if not all(np.isfinite(v).all() for v in values.values()):
                raise FloatingPointError('Non-finite analytic result')
            # Both routes should have the same effective covariance/Wick baseline.
            scale=np.maximum(np.abs(values['sigma2Wick']),eps)
            values['wickConsistencyRelError']=np.abs(values['wickFromHop']-values['sigma2Wick'])/scale
            _atomic_npz(analytic_path,**values)
        wick_err,wick_log=_deviations(var_mc[k],values['sigma2Wick'])
        hop_err,hop_log=_deviations(var_mc[k],values['sigma2Hop'])
        valid=(values['sigma2Hop']>=0)&(values['muHop']>=0)
        valid &= values['wickConsistencyRelError']<=tol
        valid &= np.abs(values['varianceImagResidual'])<=tol*np.maximum(np.abs(values['sigma2Hop']),eps)
        valid &= np.abs(values['meanImagResidual'])<=tol*np.maximum(np.abs(values['muHop']),eps)
        frame=pd.DataFrame(dict(bankID=int(prepared.row_raw.bankID),wCandidate=int(wi[k]),
            zCandidate=zi,N_MC=int(n_mc),meanEmp=mean_mc[k],varEmp=var_mc[k],
            varEmpRaw=raw_mc[k],mcVarianceClipped=raw_mc[k]<0,
            **values,analyticValid=valid,wickErrorPct=wick_err,hopErrorPct=hop_err,
            wickAbsErrorPct=np.abs(wick_err),hopAbsErrorPct=np.abs(hop_err),
            wickLogError=wick_log,hopLogError=hop_log,
            newBetterThanWick=np.abs(hop_err)<np.abs(wick_err)))
        for name in ('scenario_BR','scenario_RU','splitID','nRIS','nT','nR','fc'):
            if name in prepared.row_raw:
                frame[name]=prepared.row_raw[name]
        if 'splitID' not in frame and 'split' in prepared.row_raw:
            frame['splitID']=str(prepared.row_raw['split'])
        for j,name in enumerate(('i11','i12','i2')):
            frame[name]=int(_numpy(prepared.WIdx)[wi[k],j])
        records.append(frame)
        if progress:
            progress(f"bank {int(prepared.row_raw.bankID)} analytic W {k+1}/{len(wi)} | {len(zi)} Z")
    result=pd.concat(records,ignore_index=True)
    _atomic_table(result,result_path)
    return result


def summarize_comparisons(frame):
    """Paired errors on finite, positive-MC, valid-analytic rows only.

    Error statistics are descriptive, not CIs over millions of correlated W/Z.
    Invalid/zero-MC counts are explicit rather than silently dropped.
    """
    valid=frame.analyticValid & (frame.varEmp>0)
    valid &= np.isfinite(frame[['varEmp','sigma2Wick','sigma2Hop']]).all(axis=1)
    d=frame.loc[valid]
    def median(col):
        return float(d[col].median()) if len(d) else float('nan')
    return dict(rows=len(frame),validRows=len(d),invalidRows=int((~valid).sum()),
                banks=frame.bankID.nunique(),
                wickMedianAbsErrorPct=median('wickAbsErrorPct'),
                hopMedianAbsErrorPct=median('hopAbsErrorPct'),
                wickRMSELog=float(np.sqrt(np.nanmean(d.wickLogError**2))) if len(d) else float('nan'),
                hopRMSELog=float(np.sqrt(np.nanmean(d.hopLogError**2))) if len(d) else float('nan'),
                newBetterPct=100*float(d.newBetterThanWick.mean()) if len(d) else float('nan'))
