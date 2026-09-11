"""Existing-selector integration checks, with controlled pilot channels."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
class SelectedPrecoderTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        import torch
        from types import SimpleNamespace
        from ris_env.codebook import generate_codebook_rank1
        from ris_env.teacher_pipeline import TeacherBankPrepared
        self.torch=torch
        self.cb=generate_codebook_rank1(2,2,1,1,1,device='cpu',parity=True)
        # Known rank-one channel whose leading right singular vector is an
        # exact codeword not in our two-entry baseline pool.
        self.target=self.cb.values[:,3,2,0]
        F=torch.tensor([1.,0.5j],dtype=torch.complex128)[:,None]*self.target.conj()[None,:]
        self.H=torch.eye(4,dtype=torch.complex128)[None,:,:]
        self.G=F[None,:,:]
        row=pd.Series(dict(bankID=7,ch_seed=123,nT1=2,nT2=1))
        self.p=TeacherBankPrepared(
            row_raw=row.copy(),row_physical=row.copy(),
            static_env=SimpleNamespace(muBR=torch.zeros((4,4),dtype=torch.complex128)),
            W=torch.stack([self.cb.values[:,0,0,0],self.cb.values[:,1,0,0]]),
            WIdx=torch.tensor([[1,1,1],[1,1,2]]),
            z=torch.zeros((8,4),dtype=torch.long),
            gamma=torch.ones((8,4),dtype=torch.complex128),
            z_candidate_type=np.array(['anchor']*4+['global_random']*4),
            z_anchor_index=np.full(8,-1),z_seed_indices=np.arange(4),deterministic_timings={})
    def test_existing_selector_finds_exact_codeword_and_preserves_z_pool(self):
        from unittest.mock import patch
        from ris_env.selected_precoder_variance import select_one_precoder
        with patch('ris_env.selected_precoder_variance.row_to_link_configs',return_value=('BR','RU')), \
             patch('ris_env.selected_precoder_variance.generate_native_link_chunk',side_effect=[self.H,self.G]) as channel:
            result,meta,arrays=select_one_precoder(self.p)
        self.assertEqual(result.WIdx.tolist(),[[3,1,4]])
        self.torch.testing.assert_close(result.W[0],self.target)
        self.assertIs(result.gamma,self.p.gamma)
        self.assertIs(result.z,self.p.z)
        self.assertEqual(tuple(result.W.shape),(1,4))
        self.assertEqual(tuple(self.p.W.shape),(2,4))
        self.assertEqual(meta['selectedOriginalWCandidate'],-1)
        self.assertNotEqual(meta['selectionSeedBR'],123+70_000_121)
        self.assertNotEqual(meta['selectionSeedRU'],123+80_000_147)
        self.assertEqual(channel.call_count,2)
        self.assertAlmostEqual(meta['selectionPilotPower'],1.25,places=12)
        self.assertEqual(arrays['F_pilot'].shape,(2,4))
    def test_invalid_reference_stops_before_channel_generation(self):
        from unittest.mock import patch
        from ris_env.selected_precoder_variance import select_one_precoder
        with patch('ris_env.selected_precoder_variance.generate_native_link_chunk') as channel:
            with self.assertRaises(ValueError):select_one_precoder(self.p,reference_z_index=8)
            channel.assert_not_called()
    def test_zero_pilot_rejected(self):
        from unittest.mock import patch
        from ris_env.selected_precoder_variance import select_one_precoder
        with patch('ris_env.selected_precoder_variance.row_to_link_configs',return_value=('BR','RU')), \
             patch('ris_env.selected_precoder_variance.generate_native_link_chunk',side_effect=[self.H,self.G*0]):
            with self.assertRaises(FloatingPointError):select_one_precoder(self.p)


if __name__=='__main__':unittest.main()
