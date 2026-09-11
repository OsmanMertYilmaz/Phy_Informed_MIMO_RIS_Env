"""Independent algebra oracle; runnable without a GPU or torch install."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

# Load the pure algebra module without ris_env.__init__ GPU dependencies.
path=Path(__file__).parents[1]/'src/ris_env/hop_power_variance.py'
spec=importlib.util.spec_from_file_location('hop_algebra',path)
hop=importlib.util.module_from_spec(spec);spec.loader.exec_module(hop)


def fourth(mu,cov,a,c,b,d):
    """E[z_a z_c conj(z_b) conj(z_d)] via independent proper-Gaussian pairs."""
    return (mu[a]*mu[c]*mu[b].conjugate()*mu[d].conjugate()
        +cov[a,b]*mu[c]*mu[d].conjugate()+cov[a,d]*mu[c]*mu[b].conjugate()
        +cov[c,b]*mu[a]*mu[d].conjugate()+cov[c,d]*mu[a]*mu[b].conjugate()
        +cov[a,b]*cov[c,d]+cov[a,d]*cov[c,b])


def oracle(mx,cx,mg,kg,g):
    nr,n=mg.shape;mean=0j;second=0j;mu=mg.reshape(-1)
    for r in range(nr):
        for i in range(n):
            for j in range(n):
                mean+=g[i]*g[j].conjugate()*(cx[i,j]+mx[i]*mx[j].conjugate())*(kg[r*n+i,r*n+j]+mg[r,i]*mg[r,j].conjugate())
                for s in range(nr):
                    for k in range(n):
                        for l in range(n):
                            second+=(g[i]*g[j].conjugate()*g[k]*g[l].conjugate()
                                *fourth(mx,cx,i,k,j,l)*fourth(mu,kg,r*n+i,s*n+k,r*n+j,s*n+l))
    return mean.real,(second-mean**2).real


class HopVarianceTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(411)
    def complex(self,shape):
        return self.rng.normal(size=shape)+1j*self.rng.normal(size=shape)
    def case(self):
        n,nr=3,2
        mx=self.complex(n);mg=self.complex((nr,n))
        a=self.complex((n,n));b=self.complex((n*nr,n*nr))
        return mx,a@a.conj().T,mg,b@b.conj().T,self.complex((4,n))
    def test_full_complex_cross_covariance_against_four_index_oracle(self):
        mx,cx,mg,kg,gs=self.case();n=len(mx)
        blocks=lambda r,s:kg[r*n:(r+1)*n,s*n:(s+1)*n]
        means,variances,diag=hop.power_variance_batch(mx,cx,mg,blocks,gs)
        for j,g in enumerate(gs):
            expected=oracle(mx,cx,mg,kg,g)
            scalar=hop.power_variance_from_hop_moments(mx,cx,mg,blocks,g)
            np.testing.assert_allclose([means[j],variances[j]],expected,rtol=3e-13)
            np.testing.assert_allclose([scalar[0],scalar[1]],expected,rtol=3e-13)
            for key in diag:
                np.testing.assert_allclose(diag[key][j],scalar[2][key],rtol=2e-12,atol=1e-9)
    def test_scalar_product_not_effective_gaussian(self):
        # x~CN(0,a), g~CN(0,b): E[Y]=ab and Var(Y)=3 a^2 b^2.
        m,v,d=hop.power_variance_from_hop_moments(np.zeros(1),np.array([[2.]]),np.zeros((1,1)),[[np.array([[3.]])]],np.ones(1))
        self.assertAlmostEqual(m,6);self.assertAlmostEqual(v,108)
        self.assertAlmostEqual(d['varWickEffectiveGaussian'],36)
    def test_deterministic_hop_and_zero_gamma(self):
        mx,cx,mg,kg,gs=self.case();n=len(mx)
        zero=lambda r,s:np.zeros((n,n),complex)
        m,v,d=hop.power_variance_batch(mx,cx,mg,zero,gs)
        np.testing.assert_allclose(v,d['varWickEffectiveGaussian'],rtol=1e-13)
        m,v,d=hop.power_variance_batch(mx,cx,mg,zero,gs*0)
        np.testing.assert_array_equal(m,0);np.testing.assert_array_equal(v,0)
    def test_global_phase_and_receiver_basis_invariance(self):
        mx,cx,mg,kg,gs=self.case();n=len(mx)
        blocks=lambda r,s:kg[r*n:(r+1)*n,s*n:(s+1)*n]
        m,v,_=hop.power_variance_batch(mx,cx,mg,blocks,gs)
        unitary=np.linalg.qr(self.complex((2,2)))[0]
        transform=np.kron(unitary,np.eye(n));newkg=transform@kg@transform.conj().T
        blocks2=lambda r,s:newkg[r*n:(r+1)*n,s*n:(s+1)*n]
        m2,v2,_=hop.power_variance_batch(mx,cx,unitary@mg,blocks2,gs*np.exp(0.73j))
        np.testing.assert_allclose(m,m2,rtol=1e-13);np.testing.assert_allclose(v,v2,rtol=1e-13)
    def test_input_errors(self):
        mx,cx,mg,kg,gs=self.case()
        with self.assertRaises(ValueError):hop.power_variance_batch(mx,cx,mg,[],gs[:,:1])
        gs[0,0]=np.nan
        with self.assertRaises(ValueError):hop.power_variance_batch(mx,cx,mg,[],gs)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
    def test_torch_batch_matches_numpy(self):
        import torch
        mx,cx,mg,kg,gs=self.case();n=len(mx)
        cb=lambda r,s:kg[r*n:(r+1)*n,s*n:(s+1)*n]
        expected=hop.power_variance_batch(mx,cx,mg,cb,gs)
        tcb=lambda r,s:torch.tensor(cb(r,s))
        actual=hop.power_variance_batch(*map(torch.tensor,(mx,cx,mg)),tcb,torch.tensor(gs))
        np.testing.assert_allclose(actual[0].numpy(),expected[0],rtol=1e-13)
        np.testing.assert_allclose(actual[1].numpy(),expected[1],rtol=1e-13)


if __name__=='__main__':unittest.main()
