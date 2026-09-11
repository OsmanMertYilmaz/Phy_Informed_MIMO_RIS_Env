# MC / Wick / conditional-hop variance experiment

Run `notebooks/compare_mc_wick_new_analytic_4000.ipynb` against the existing
`environments/environments_4000.csv`. It uses the repository's existing
`prepare_teacher_bank` candidate pool, `stateful_mc` native channel generator,
and `snr_statistics.evaluate_gamma_batch` Wick calculation. No MATLAB input,
MATLAB runtime, MC moment fitting, or L1/L2/L3 model is used.

Defaults evaluate all 4000 CSV banks, 32 W × 512 Z per bank, and 64,000 native
channel pairs per bank. CSV split labels are retained. Candidate selection is
deterministic and completed before the evaluation MC. All three estimates
use the identical saved W and gamma. `BANK_IDS`, `W_INDICES`, and `Z_INDICES`
permit an explicitly selected pilot; `None` means all, with no implicit cap.

## New calculation

`hop_power_variance.power_variance_from_hop_moments` is a scalar NumPy/torch
reference; `power_variance_batch` evaluates several gamma candidates together.
For x=Hw, y=diag(gamma)x, and P=||Gy||², it evaluates

    Var(P) = Var(E[P|y]) + E[Var(P|y)]
           = varConditionalMean + expectedTraceV2 + 2 expectedMeanVMean.

The hop means and centered covariances come only from the existing analytic
environment. All receive cross-covariance blocks are retained. The B matrix
in the noncentral quadratic-product term is rank one, allowing an algebraic
reduction of that part's matrix products. The reference and batched formulas
are exact under independent proper complex Gaussian hops. They do not assume
the effective cascaded channel is Gaussian. The native finite-ray/XPR channel
is the separate MC benchmark, so agreement with it is an empirical question.

The dense new calculation costs O(C nR² nRIS³) per W; it is substantially more
expensive than Wick. `analytic_z_chunk` controls temporary memory, not the
number of candidates. The default double precision can be changed explicitly,
using a new output directory. No faster diagonal-only approximation is used.

## Output / resume

Each bank directory contains:

- `manifest.json`: input row, candidate hash, settings, device, torch version,
  and source hashes. Incompatible restarts fail before combining results.
- `candidates.npz`: actual W, gamma, z, WIdx and candidate indices.
- `mc_checkpoint.npz`: both RNG states, sum(Y), sum(Y²), and sample count.
- `mc_at_*.npz`: convergence snapshots of the same stream.
- `analytic_w_*.npz`: new and old analytic results after each completed W.
- `comparison.parquet`: `varEmp`, `sigma2Wick`, `sigma2Hop`, means, signed and
  absolute percent errors, log errors, decomposition and validity diagnostics.

MC accumulation and Wick behavior are unchanged. `varEmpRaw` and
`mcVarianceClipped` expose cancellation/clipping in the existing MC engine.
The new analytic result is not clipped. Its imaginary residual and agreement
of its reconstructed Wick baseline with the existing Wick API are recorded;
invalid analytic rows remain in the table and are excluded explicitly from
paired summary metrics. Zero-MC-variance relative errors are undefined.

The notebook updates bank summaries, a bounded plotting sample, and figures.
The overview reports distributions of **bank-level** errors, not global
confidence intervals from correlated W/Z rows. Convergence snapshots also
are not independent replicates. Use only one writer per output directory.
Existing teacher datasets and learned models are not overwritten.

## Verification

`python tests/test_hop_power_variance.py -v` checks the NumPy algebra against
an independent four-index Gaussian-moment expansion, the scalar product
variance (3× the effective-Gaussian Wick value at zero mean), deterministic
hop limits, zero RIS response, global-phase invariance, and receiver-unitary
invariance. A torch equivalence check runs when torch is installed.

Development verification in this session: 5 NumPy tests passed; torch test
skipped because torch was unavailable. Notebook JSON and Python cell syntax
were checked. The full native MC/GPU run was not executed in this session.
No numerical advantage over Wick is claimed before that experiment runs.
