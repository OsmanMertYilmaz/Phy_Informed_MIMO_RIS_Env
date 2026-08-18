# Validation Status

All production-name modules originate from numerical implementations that
were validated against the MATLAB reference pipeline before this refactor.

## Closed stages

| Scope | Status | Key result |
|---|---|---|
| Antenna/channel moments | PASS | double precision ~ machine precision |
| Spatial rho / GH | PASS | double ~1e-14, float32 ~1e-6 |
| Effective moments / Wick | PASS | double ~1e-14, float32 ~1e-6 |
| RIS response | PASS | double exact; float32 ~5e-8 |
| Type-I rank-1 codebook | PASS | indices 100% match |
| Geometry + LSP | PASS | double ~1e-15; float32 ~3e-7 |
| Deterministic environment | PASS | double ~6e-15; float32 ~5e-6 |
| Fixed stochastic primitives | PASS | double ~1.7e-14; float32 ~6.7e-6 |
| Native CUDA stochastic stats | PASS | old ~10k MATLAB mean/variance agree with Python 100k |
| Symmetric-GG N convergence | PASS | production N_MC = 64,000 |
| Multi-W / multi-z label engine | PASS | 32W x 512z, 64k MC benchmark completed |

## B3-B benchmark snapshot

Heavy benchmark bank: nT=16, nR=8, nRIS=512.

- 16,384 final labels
- engine total ~8.893 s
- ~1,842 Symmetric-GG labels/s
- ~121.55M candidate-sample evaluations/s
- peak CUDA allocation ~1.685 GB
- lookup clamped: 0%
- analytic mean vs 64k empirical mean: MdAPE ~0.386%, P90 ~0.723%

## Important conventions

- Dual-pol order and physical array layout must remain MATLAB-compatible.
- RIS binary levels are 45° and 135° with phase-dependent amplitude.
- Type-I scope: XP=2, cb_mode=1, nl=1.
- Population empirical variance: `E[Y^2]-E[Y]^2` / MATLAB `var(Y,1)`.
- Production distribution target: Symmetric Gamma-Gamma q05.
- Do not replace analytic mean with empirical mean in production labels.

Historical notebooks in `notebooks/validation/` contain the detailed parity
workflows and should be retained when equations are changed.
