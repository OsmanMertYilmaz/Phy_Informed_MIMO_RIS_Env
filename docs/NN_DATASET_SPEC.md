# NN Dataset Specification — v1

## Unit of observation

One row is one physical candidate:

```text
(environment x, precoder W, RIS pattern z)
```

## Locked scale

- 4,000 banks
- 32 W per bank
- 512 z per bank
- 64,000 Monte-Carlo realization pairs per bank
- 16,384 candidates per bank
- 65,536,000 total candidates

## Environment distribution

Each scenario family has 1,000 banks:

| Family | LOS/LOS | LOS/NLOS | NLOS/LOS | NLOS/NLOS | Total |
|---|---:|---:|---:|---:|---:|
| Indoor-Office | 250 | 250 | 250 | 250 | 1000 |
| UMi | 250 | 250 | 250 | 250 | 1000 |
| UMa | 250 | 250 | 250 | 250 | 1000 |
| RMa | 250 | 250 | 250 | 250 | 1000 |

`nRIS` is balanced over `{64,128,256,512}` with approximately 1,000 banks
per size. `nT >= 4` is required so 32 unique Type-I rank-1 W candidates are
available consistently.

Splits are bank-level: 2,800 train / 600 validation / 600 controlled
geometry-interpolation test. No W/z candidate from a held-out bank may leak
into training.

## 512 RIS-pattern distribution

- 4 anchors: all-0, all-1, alternating 0101..., alternating 1010...
- 256 global i.i.d. Bernoulli(0.5) patterns
- 96 phase-density-stratified random patterns
  - target 1-ratios: 0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9
  - 12 patterns per density
- 64 structured 2-D spatial patterns
  - horizontal/vertical stripes
  - checkerboards
  - blocks / periodic masks
- 92 local perturbations
  - 4 analytically strong seed patterns
  - 23 perturbations per seed
  - Hamming perturbation sizes drawn from {1,2,4,8}

All 512 patterns must be unique; duplicates are replaced.

## 32 W distribution

- 8 unique W selected from an independent pilot realization stream
- 24 unique Type-I rank-1 codewords selected to cover the remaining codebook
- duplicate pilot selections are filled by unused codewords
- pilot-selection RNG must be independent of the 64k label MC stream

## Label

The production target is **not** raw empirical q05.

```text
muSNR_analytic + varEmp_64k -> Symmetric Gamma-Gamma -> q05GG
target = log(q05GG)
```

MC variance uses the population convention (`var(Y,1)` in MATLAB):

```text
varEmp = E[Y^2] - E[Y]^2
```

## Stored diagnostics (target-only, never NN inputs)

`meanEmp64k`, `varEmp64k`, `varRatio64k`, `logVarRatio64k`, `ggShapeA`,
`q05GG`, `logQ05GG`.

## Actor path

After the q05 scorer is trained, multi-start bit-flip / multi-bit searches
are run on the scorer. High-scoring final patterns are re-evaluated with
the 64k-MC physics engine. Verified top patterns become Actor teacher labels:

```text
(environment, W) -> {z*_1, z*_2, ...}
```

A second scorer training pass should include verified high-q05 search-region
samples before final Actor-teacher generation.
