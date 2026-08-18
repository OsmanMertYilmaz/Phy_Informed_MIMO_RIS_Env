# Phy_Informed_MIMO_RIS_Env

GPU-native, MATLAB-parity-validated physics environment for RIS-assisted
MIMO statistical modeling and **Symmetric Gamma-Gamma q05** label generation.

## Current status

The numerical port through the stochastic multi-W / multi-z label engine is
validated. The production dataset design is locked at:

- 4,000 independent environments/banks
- 32 unique Type-I rank-1 precoders per bank
- 512 RIS binary phase patterns per bank
- 64,000 Monte-Carlo BR/RU realization pairs per bank
- 16,384 labels per bank
- 65,536,000 total candidate labels
- target: `log(q05GG)`

The statistical label is

```text
analytic muSNR + MC varEmp -> Symmetric Gamma-Gamma -> q05GG
```

Raw empirical 5th-percentile labels are **not** the production target.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

CUDA-enabled PyTorch must match the host CUDA stack. Install the appropriate
PyTorch wheel first when necessary.

## Package map

```text
src/ris_env/
  antenna.py              array geometry + channel moments
  geometry_lsp.py         geometry + LSP generation
  spatial_correlation.py  second-order / Gauss-Hermite rho
  ris_response.py         binary RIS phase/amplitude response
  codebook.py             Type-I rank-1 codebook + selection
  snr_statistics.py       effective moments, C, muSNR, Wick variance
  environment.py          deterministic bank assembly
  rho_cache.py            same-shape rho caching/batching
  channel_primitives.py   fixed-random-primitives stochastic channel core
  channel_realizations.py native PyTorch/CUDA stochastic RNG
  validation.py           legacy-dataset cross-validation helpers
  gamma_gamma.py          symmetric-GG fit/lookup + N-convergence helpers
  label_engine.py         multi-W/multi-z 64k-MC streaming label engine
```

Production modules deliberately preserve equations and conventions that
passed MATLAB parity. See `docs/VALIDATION_STATUS.md` before refactoring
internals.

## Dataset design

See:

- `configs/nn_dataset_4000.yaml`
- `docs/NN_DATASET_SPEC.md`

The 512 RIS patterns per bank are intentionally mixed rather than purely
random: anchors, global Bernoulli patterns, phase-density-stratified
patterns, structured 2-D patterns, and local Hamming perturbations.

## Next milestones

1. Implement controlled geometry-interpolation scenario generator for 4,000 banks.
2. Wire the production dataset writer/checkpointing pipeline.
3. Generate the 65.536M-row Symmetric-GG q05 dataset.
4. Train direct `log(q05GG)` NN scorer.
5. Search high-q05 RIS patterns with the scorer and verify finalists with 64k MC.
6. Retrain scorer on high-q05 regions and generate Actor teacher data.
7. Train Actor `(environment, W) -> z`.

## Reproducibility

Historical parity notebooks and MATLAB exporters are kept under
`notebooks/validation/` and `tests/matlab_golden/exporters/`. These are
validation provenance, not production APIs.
