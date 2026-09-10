# Level-3 generalization v2

Notebook: `notebooks/train_4000_level3_generalization_v2.ipynb`.
Derived from the existing `train_4000_channel_map_level3.ipynb`; this is not
a new teacher pipeline. The notebook checks out `codex/l3-generalization-v2`
in a separate Colab code directory and uses the existing Drive dataset.

## Default run

```python
Z_INPUTS = ["row_phys", "pattern", "interaction"]
L3_RUN_TAG = "interaction_fixed_v2"
L3_USE_LEVEL2_CONTEXT = False
L3_LAMBDA_ZERO = 0.0
FORCE_RETRAIN = False
```

Run cells in order. Existing Level-1/2 models and interaction sidecars are
reused. `FORCE_RETRAIN=False` does not suppress a new L3 run when its checkpoint
does not exist. Keep this flag false to avoid retraining Level-1/2.

Changes:

- A common `physical_ris_grid` maps port vectors to `[polarization, N, M]`.
  M is the fastest port index. Spatial/polarization/FFT pattern descriptors,
  raw-pattern CNN and channel-map phase pooling now use this convention.
- The 13 corrected descriptors are saved atomically as
  `level3/z_pattern_features_physical_v2.npy`. Their cache checks the packed
  patterns and bank table using SHA-256. The previous descriptor file remains
  available, and valid existing Level-3 row shards are reused.
- Pattern normalization fits training banks and training Z candidates only.
- `level2DeltaHat` is excluded from the L3 context by default. Level-2's
  prediction still contributes to the end-to-end estimate. Its old training
  predictions are in-sample, so they should not be restored as L3 inputs without
  a deliberate ablation or bank-grouped out-of-fold prediction generation.
- Zero-mean regularization is disabled for randomly sampled subsets of 48 Z.
  The residual's exact zero-mean property holds over the full 358 training Z,
  not over each subset. MSE and centered MSE remain.
- Checkpoints include the scalar list, physical ordering version, normalization,
  feature lists, split/data fingerprints and training configuration. Incompatible
  checkpoints are not silently loaded. Use a distinct L3 run tag for each ablation.
- Channel-map extraction is optional. It no longer runs in the default setup.

The L3 target remains `targetLogVarRatio - targetWMean`. No teacher label,
Monte Carlo statistic or reference q05 is added to deployment inputs.

## Optional full covariance ablation

```python
Z_INPUTS = ["row_phys", "pattern", "interaction", "covariance"]
L3_RUN_TAG = "interaction_covariance_v2"
```

The optional cell computes eight analytic features, stores atomic/resumable
Parquet sidecars in `hierarchical_teacher/z_covariance_v1`, and checks source
and environment hashes when resuming. It reads candidate keys and `zString`
from teacher shards, never target or MC columns. L3 joins the sidecars by
`bankID`, W codebook indices and `zCandidate`, with duplicate and missing-key
checks. It does not rewrite the existing Level-3 row shards.

Component names always specify **RU first, BR second**:

| Name | RU | BR |
| --- | --- | --- |
| LL | LOS | LOS |
| LN | LOS | NLOS |
| NL | NLOS | LOS |
| NN | NLOS | NLOS |

With `U_L = ubar ubar^H`, `U_N = UBR - U_L`,
`R_L = sum_r muRU_r muRU_r^H`, and `R_N = nR * rho_RUhop`, define
`K_ab = R_a .* U_b`. Powers use the project's unconjugated-left convention:
`P_ab = real(gamma^T K_ab gamma*)`. Diagonal contributions are
`D_ab = sum_i |gamma_i|^2 K_ab[i,i]`; RIS response amplitudes are not assumed
equal between states.

Four features are `P_ab / sum(P)`. Four signed features are
`(P_ab - D_ab) / sum(D)`. The shared diagonal denominator avoids division by
an absent LOS component. Covariance ratios stay in linear space during input
normalization. Computation batches Z candidates and uses no new Monte Carlo.
These are second-moment features, not a claim of an exact variance correction.

## Evaluation and verification scope

Reference interaction validation RMSE: newGeom `0.101109`, newBoth `0.101057`.
Compare the corrected default first, then add covariance. Repeat promising
comparisons across seeds before drawing conclusions. `newBoth` is used for
model selection and remains validation, not an untouched final test.

Code preparation checks JSON, Python syntax and notebook structure. No training,
test suite, teacher generation, real-bank numerical run or A100 benchmark is
executed during this change. Improvements and runtime must be measured in Colab.
