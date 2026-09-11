"""Conditional-variance port of power_variance_from_hop_moments.m.

Exact for independent proper complex Gaussian hops. Native finite-ray/XPR
hops need a separate MC comparison. This does not assume a Gaussian cascade.
NumPy and torch inputs follow the same operations; torch enables CUDA use.
"""
from __future__ import annotations

import numpy as np

METHOD_VERSION = "conditional_hop_variance_v1"


def power_variance_from_hop_moments(mu_x, cx, mu_g, cg, gamma):
    """Return mean power, power variance, and decomposition diagnostics.

    mu_x [I], cx [I,I], mu_g [R,I], gamma [I]. cg is a callable (r,s)
    or a nested sequence of covariance blocks [I,I], with
    cg(r,s)[i,j] = E[(G[r,i]-mu_g[r,i])*conj(G[s,j]-mu_g[s,j])].
    Indices are zero-based. All receive cross-blocks are required.
    cg blocks are transposed WITHOUT conjugation at the indicated steps.
    Negative outputs and imaginary residuals are reported, never clamped.
    Dense cost per gamma: O(R**2 * I**3); working space O(I**2) besides inputs.
    """
    is_torch = hasattr(mu_g, "detach")
    if is_torch:
        import torch
        dev = mu_g.device
        inputs = [torch.as_tensor(x,device=dev) for x in (mu_x,cx,mu_g,gamma)]
        dtype = torch.complex128 if any(x.dtype in (torch.float64,torch.complex128) for x in inputs) else torch.complex64
        mu_x,cx,mu_g,gamma = [x.to(dtype) for x in inputs]
        finite = lambda x: bool(torch.isfinite(x).all())
        zeros = lambda shape: mu_g.new_zeros(shape)
    else:
        mu_x, cx, mu_g, gamma = map(np.asarray, (mu_x, cx, mu_g, gamma))
        finite = lambda x: bool(np.isfinite(x).all())
        zeros = lambda shape: np.zeros(shape, dtype=np.result_type(mu_g, cx, complex))
    if mu_g.ndim != 2:
        raise ValueError("mu_g must be nonempty [R,I]")
    nr, n = mu_g.shape
    mu_x, gamma = mu_x.reshape(-1), gamma.reshape(-1)
    if nr < 1 or n < 1 or cx.shape != (n, n) or mu_x.shape != (n,) or gamma.shape != (n,):
        raise ValueError("Hop moment shape mismatch")
    if not all(finite(x) for x in (mu_x, cx, mu_g, gamma)):
        raise ValueError("Non-finite hop moments or RIS response")

    def block(r, s):
        k = cg(r, s) if callable(cg) else cg[r][s]
        k = torch.as_tensor(k,device=dev,dtype=dtype) if is_torch else np.asarray(k)
        if k.shape != (n, n) or not finite(k):
            raise ValueError(f"Invalid RU covariance block ({r},{s})")
        return k

    trace = lambda a: a.diagonal().sum()
    m = gamma * mu_x
    S = (gamma[:, None] * gamma.conj()[None, :]) * cx
    T = mu_g.conj().T @ mu_g
    for r in range(nr):
        T = T + block(r, r).T
    TS = T @ S
    mean_raw = trace(TS) + m.conj() @ T @ m
    var_conditional_mean = (TS * TS.T).sum() + 2 * (m.conj() @ TS) @ (T @ m)

    def quadratic_product(A, B, AS):
        BS = B @ S
        Am, Bm = A @ m, B @ m
        return ((trace(AS) + m.conj() @ Am) * (trace(BS) + m.conj() @ Bm)
                + (AS * BS.T).sum()
                + (m.conj() @ AS) @ Bm + (m.conj() @ BS) @ Am)

    expected_trace_v2 = 0
    expected_mean_v_mean = 0
    vbar = zeros((nr, nr))
    for r in range(nr):
        for s in range(nr):
            A = block(r, s).T
            B = mu_g[r, :].conj()[:, None] * mu_g[s, :][None, :]
            AS = A @ S
            vbar[r, s] = trace(AS) + m.conj() @ A @ m
            expected_trace_v2 += quadratic_product(A, A.conj().T, AS)
            expected_mean_v_mean += quadratic_product(A, B, AS)
    raw_variance = var_conditional_mean + expected_trace_v2 + 2 * expected_mean_v_mean
    mean_f = mu_g @ m
    cf = mu_g @ S @ mu_g.conj().T + vbar
    wick = ((cf * cf.T).sum() + 2 * mean_f.conj() @ cf @ mean_f).real
    return mean_raw.real, raw_variance.real, {
        "varWickEffectiveGaussian": wick,
        "varianceImagResidual": raw_variance.imag,
        "meanImagResidual": mean_raw.imag,
        "varConditionalMean": var_conditional_mean.real,
        "expectedTraceV2": expected_trace_v2.real,
        "expectedMeanVMean": expected_mean_v_mean.real,
        "CF": cf,
        "meanF": mean_f,
    }


def prepare_ru_covariance_blocks(env):
    """All RU covariance blocks from the existing analytic ray moments.

    The formulas match snr_statistics.prepare_w_state before multiplying UBR.
    No diagonal-only approximation, empirical moments, or MC samples are used.
    """
    import torch
    nr, n = env.muRU.shape
    count = env.rho_RU.shape[2]
    signs = env.sRIS[:, None] * env.sRIS[None, :]
    c0 = env.sigma2RU / (2 * (1 + env.eKappaRU) * count)
    blocks = [[torch.zeros((n, n), dtype=env.muRU.dtype, device=env.muRU.device)
               for _ in range(nr)] for _ in range(nr)]
    for ell in range(count):
        gt = env.rho_UR[:, :, ell]
        gp = gt * signs
        direct = gt + env.eKappaRU * gp
        cross = env.eKappaRU * gt + gp
        for r in range(nr):
            for s in range(nr):
                blocks[r][s] += (c0 * env.rho_RU[r, s, ell]
                                 * (direct + env.sR[r] * env.sR[s] * cross))
    return blocks


def prepare_br_moments(env, w):
    """Analytic mean and centered covariance of x=H*w (no mean subtraction)."""
    import torch
    w = torch.as_tensor(w, dtype=env.muBR.dtype, device=env.muBR.device).reshape(-1)
    n, nt = env.muBR.shape
    if w.numel() != nt:
        raise ValueError("Precoder dimension mismatch")
    sr = env.sRIS[:, None] * env.sRIS[None, :]
    st = env.sT[:, None] * env.sT[None, :]
    count = env.rho_RB.shape[2]
    c0 = env.sigma2BR / (2 * (1 + env.eKappaBR) * count)
    cx = torch.zeros((n, n), dtype=env.muBR.dtype, device=env.muBR.device)
    for ell in range(count):
        rx, tx = env.rho_RB[:, :, ell], env.rho_BR[:, :, ell]
        s1 = ((w @ tx) * w.conj()).sum()
        s2 = ((w @ (tx * st)) * w.conj()).sum()
        rp = rx * sr
        cx += c0 * ((rx + env.eKappaBR * rp) * s1
                    + (env.eKappaBR * rx + rp) * s2)
    return env.muBR @ w, cx


def power_variance_batch(mu_x, cx, mu_g, cg, gamma):
    """Candidate-batched conditional variance, NumPy or torch/CUDA.

    gamma [C,I]. Same equations as the scalar reference above. Uses the
    rank-one structure of B=conj(mu_g[r]) outer mu_g[s] to avoid a dense
    B@S multiplication. No Gaussian effective-channel approximation.
    Cost O(C R^2 I^3); choose a small external candidate chunk (default 4).
    """
    is_torch = hasattr(mu_g, "detach")
    if is_torch:
        import torch
        dev = mu_g.device
        inputs = [torch.as_tensor(x,device=dev) for x in (mu_x,cx,mu_g,gamma)]
        dtype = torch.complex128 if any(x.dtype in (torch.float64,torch.complex128) for x in inputs) else torch.complex64
        mu_x,cx,mu_g,gamma = [x.to(dtype) for x in inputs]
        finite = lambda x: bool(torch.isfinite(x).all())
        zeros = lambda shape: mu_g.new_zeros(shape)
    else:
        mu_x, cx, mu_g, gamma = map(np.asarray, (mu_x, cx, mu_g, gamma))
        finite = lambda x: bool(np.isfinite(x).all())
        zeros = lambda shape: np.zeros(shape, dtype=np.result_type(mu_g, cx, complex))
    if mu_g.ndim != 2 or gamma.ndim != 2:
        raise ValueError("mu_g must be [R,I] and gamma [C,I]")
    nr, n = mu_g.shape
    c = gamma.shape[0]
    mu_x = mu_x.reshape(-1)
    if min(nr, n, c) < 1 or gamma.shape[1] != n or cx.shape != (n,n) or mu_x.shape != (n,):
        raise ValueError("Hop moment shape mismatch")
    if not all(finite(x) for x in (mu_x, cx, mu_g, gamma)):
        raise ValueError("Non-finite hop moments or RIS response")

    def block(r,s):
        k = cg(r,s) if callable(cg) else cg[r][s]
        k = torch.as_tensor(k,device=dev,dtype=dtype) if is_torch else np.asarray(k)
        if k.shape != (n,n) or not finite(k):
            raise ValueError(f"Invalid RU covariance block ({r},{s})")
        return k

    tr = lambda a: a.diagonal(dim1=-2,dim2=-1).sum(-1) if is_torch else np.trace(a,axis1=-2,axis2=-1)
    mt = lambda a: a.swapaxes(-1,-2)
    dot = lambda a,b: (a*b).sum(-1)
    matvec = lambda a,v: (a @ v[...,None])[...,0]
    m = gamma*mu_x
    mh = m.conj()
    S = gamma[:,:,None]*gamma.conj()[:,None,:]*cx
    T = mu_g.conj().T@mu_g
    for r in range(nr):
        T = T + block(r,r).T
    TS = T@S
    mean = tr(TS)+dot(mh,matvec(T,m))
    vcm = (TS*mt(TS)).sum((-1,-2))+2*dot(matvec(mt(TS),mh),matvec(T,m))
    ev2 = zeros((c,))
    emvm = zeros((c,))
    vbar = zeros((c,nr,nr))
    for r in range(nr):
        for s in range(nr):
            A = block(r,s).T
            AS = A@S
            Am = matvec(A,m)
            qa = tr(AS)+dot(mh,Am)
            vbar[:,r,s] = qa
            AH = A.conj().T
            AHS = AS if r == s else AH@S
            AHm = matvec(AH,m)
            ev2 = ev2 + (qa*(tr(AHS)+dot(mh,AHm))
                +(AS*mt(AHS)).sum((-1,-2))
                +dot(matvec(mt(AS),mh),AHm)+dot(matvec(mt(AHS),mh),Am))
            # B=u v^T, so B@S = u (v^T S).
            u,v = mu_g[r].conj(),mu_g[s]
            vS = matvec(mt(S),v)
            um = dot(mh,u)
            vm = dot(m,v)
            qb = dot(vS,u)+um*vm
            emvm = emvm + (qa*qb+dot(vS,matvec(AS,u))
                +dot(matvec(mt(AS),mh),u)*vm+um*dot(vS,Am))
    raw = vcm+ev2+2*emvm
    mean_f = m@mu_g.T
    cf = mu_g@S@mu_g.conj().T+vbar
    wick = ((cf*mt(cf)).sum((-1,-2))+2*dot(mean_f.conj(),matvec(cf,mean_f))).real
    return mean.real,raw.real,{
        "varWickEffectiveGaussian":wick,
        "varianceImagResidual":raw.imag,"meanImagResidual":mean.imag,
        "varConditionalMean":vcm.real,"expectedTraceV2":ev2.real,
        "expectedMeanVMean":emvm.real,
    }
