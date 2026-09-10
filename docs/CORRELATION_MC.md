# Correlation formula: independent algebra and native Monte Carlo

This validation accompanies `Korelasyon(1).pdf`. The existing production
`snr_statistics.prepare_w_state` already contracts the precoder before scoring
RIS candidates. The new module independently implements the centered A+B+C
formula and checks it against both a small four-index oracle and production.
Production channel, rho, RIS response and label equations are unchanged.

## Quantities and assumptions

Let `x = H_BR @ w` and `f_r = sum_i gamma_i * G_RU[r,i] * x_i`.
`gamma` is the complete complex reflection coefficient, including amplitude
and phase exactly once. The project uses binary phases 45/135 degrees with
its existing phase-dependent amplitude. The same fixed Type-I W and RIS
patterns are used by every analytical and MC path.

For independent BR and RU links, with deterministic environment parameters:

```math
C_{z,r}=C_{g,r}\odot C_x
       +C_{g,r}\odot(\mu_x\mu_x^H)
       +C_x\odot(\mu_{g,r}\mu_{g,r}^H),
\qquad \mathrm{Var}(f_r)=\gamma^T C_{z,r}\gamma^*.
```

No Gaussian or Wick assumption is required for this identity. Its input
covariances still need to describe the same channel ensemble. The native
generator redraws capped lognormal angular spreads, cluster offsets, XPR and
shared per-ray polarization phases. BR and RU use separate seeded RNG streams.
Array ordering, direction vectors, angle wrapping and 2*pi spatial phases
come from the existing project modules.

The PDF shows a first-order angular expansion; the current repository uses
second-order angular expansion plus Gauss-Hermite averaging. This experiment
uses that current rho, and native MC retains the original trigonometry.
Agreement with the four-sum oracle verifies contraction, NOT exactness of the
angular approximation. The experiment is for this project's shared-ray phase
model, not independently drawn phases for each antenna pair.

## Four comparisons

1. **Algebra:** centered contraction vs explicit `(i,t,j,u)` second moment
   on small arrays. The size guard is checked before allocating the oracle.
2. **Existing implementation:** means, UBR and channel variances vs production.
   The independent path returns raw variance; production clips at zero.
3. **Native MC:** complex mean, channel variance, centered energy standard
   error, mean power, and empirical power variance at cumulative checkpoints.
4. **Power diagnostic:** empirical `Var(sum_r |f_r|^2)` vs existing Wick.
   This comparison is not used as an algebra correctness gate.

Channel variance and power variance must not be confused. A separate helper
`gaussian_scalar_power_variance` implements

```math
A=D^H C_g^T D,\quad
\mathrm{Var}(|g^TDx|^2)=[\mathrm{tr}(AC_x)]^2+2\mathrm{tr}(AC_xAC_x).
```

This helper is exact only for independent zero-mean proper Gaussian x and g,
and a scalar output. Its unit test uses independent complex Gaussian MC with
non-real spatial covariance. It is not applied to native finite-ray/XPR,
LOS or multi-receiver power variance. For scalar unit variances the exact
answer is 3, while a Gaussian approximation of the cascaded scalar gives 1.

## Run

```bash
pip install -e '.[dev]' matplotlib
pytest -q tests/test_correlation_validation.py

# Short CPU validation: four banks, LOS/NLOS, two RIS sizes.
python scripts/validate_correlation_mc.py \
  --nris 8 32 --n-mc 8192 --mc-chunk 128 --device cpu \
  --output outputs/correlation_mc_cpu

# Larger experiment; device auto selects CUDA when available.
python scripts/validate_correlation_mc.py \
  --nris 32 128 512 --n-mc 65536 --mc-chunk 128 \
  --output outputs/correlation_mc_64k
```

Double precision is the default. `--float32` exercises the lower-memory path.
`--cpu-threads 1` avoids CPU oversubscription. The default demo has nT=nR=4;
nRIS counts BOTH polarizations, so nRIS=512 is a 16x16 dual-pol surface.
Four phase patterns are scored on the same channel draws. Pattern 0 is all
45 degrees, 1 all 135 degrees, 2 alternating, and 3 seeded random.

For a real geometry, `--bank-json banks.json` accepts a list of dictionaries
with the `BankInput` fields: `scenario_br`, `scenario_ru`, `fc`, `ris`, `gnb`,
`ue`, `nT1`, `nT2`, `nR1`, `nR2`, `nRIS_x`, `nRIS_y`, and `WIdx` (MATLAB
one-based indices). This replaces the demo banks. See the notebook for a
ready-to-edit example. No existing dataset or MATLAB reference file is needed.

## Outputs and restart

- `manifest.json`: exact bank/configuration, RNG seed, precision, device,
  PyTorch version and source fingerprint.
- `bank_*_progress.csv`: saved at each MC checkpoint, including final N.
- `bank_*.json`: completed bank, explicit W and z, errors, timings and metrics.
- `convergence.csv`, `final_comparison.csv`, `algebra_checks.csv`.
- `comparison.png`: channel convergence and the separate power diagnostic.

Use the same command with `--resume` to skip completed banks. The manifest
must match. A partially completed bank is restarted from its seed; partial
CSV is diagnostic, not RNG state. An interruption loses at most the active
bank's work. Output is not overwritten by a new run unless it is an explicitly
compatible resume. The Colab notebook places this directory in Google Drive.

Seeds, chunk sizes and checkpoint boundaries are recorded. Changing chunk
sizes/boundaries can change the RNG sequence despite the same seed. MC
checkpoints and candidates share samples; their errors are correlated.

## Reading results

- `four_sum_*_rel_error` should be near floating-point precision. It is skipped
  at large N by design, and remains explicit in `four_sum_status`.
- `variance_rel_error` compares population MC variance (ddof=0) with raw
  analytical channel variance, per candidate and receiver.
- `centered_energy_se` estimates the standard error of the MC mean of
  `|f-analytic_mean|^2`, using its sample variance divided by N.
- `centered_residual_se` is that centered-energy discrepancy divided by SE.
  Persistent residuals over several SE motivate checking angular rho/model
  assumptions. This is a diagnostic, not a multiple-comparison hypothesis test.
- `mean_error_over_channel_std` remains meaningful when the true mean is zero.
- Negative raw analytical variance or appreciable imaginary residuals should
  be investigated. No clipping is applied to the new analytical result.
- Power metrics repeat across receiver rows; deduplicate `(bank,candidate)`
  when plotting or averaging power results.

The CLI fails on algebraic mismatch. Native Monte Carlo discrepancies are
reported without a blanket PASS claim because rho is approximate. Timings are
raw synchronized wall times, not hardware-independent speedup claims. Bank
preparation still calls the existing full production WState for comparison;
the independent contracted state itself needs only O(nR*nRIS^2) memory.
