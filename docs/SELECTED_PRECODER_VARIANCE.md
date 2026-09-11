# Selected precoder: 4000 × 1 W × 512 Z

Notebook: `notebooks/compare_selected_precoder_variance_4000.ipynb`.

This controlled follow-up asks whether the large variance errors are associated
with W candidates in the old 32-codeword pool. It uses the existing
`codebook.select_precoder_rank1` function, not a new q05/error scoring rule.

For each bank:

1. Reuse `prepare_teacher_bank(k_w=32)` to preserve the original 512 Z pool.
   Those 32 W values are used only in the existing deterministic pool preparation.
2. Generate one independent pilot pair through `generate_native_link_chunk`.
   Default pilot seeds are `ch_seed+10_000_019` and `ch_seed+20_000_033`, distinct
   from evaluation seeds `ch_seed+70_000_121` and `ch_seed+80_000_147`.
3. Fix a reference RIS pattern: index 4, the first global-random pattern after
   the four anchors. Form the pilot cascaded channel and take its SVD.
4. Call the existing selector with V and the full rank-one Type-I codebook.
   This can select a codeword outside the original evenly-spaced 32-codeword pool.
5. Freeze that one W for all 512 Z and all evaluation channel realizations.
   Evaluate MC, existing Wick, and the new conditional-hop variance through
   the unchanged `compare_prepared_bank` implementation.

This is the selector's W for one pilot/reference Z, not a guarantee of a global
optimum across all RIS patterns or future channels. Changing W per Z or per MC
sample would define a different experiment and is deliberately not done.

## Results

Default output:
`analytic_variance_comparison/mc_wick_new_selected_precoder_4000_v1/`.

Each bank has the existing resumable MC/analytic outputs, plus
`precoder_selection.json` and `selection_pilot.npz`. Actual selected W/gamma
are in `candidates.npz`. `wCandidate=0` is a local index; identify the physical
codeword with `i11/i12/i2`. `selectedOriginalWCandidate=-1` means the selector's
codeword was not in the previous 32-W pool.

`bank_summary.csv` includes median/P90/P95/P99 errors and the fraction above
20%, all computed from the selected W's complete 512-Z evaluation (unless the
user explicitly chooses a smaller Z subset).

`BASELINE_ROOT` optionally points to completed banks from the previous 32-W
run. Physical row and Z/gamma identity are checked before comparing results.
`baseline_comparison.csv` compares errors for the full pool and the selected W.
When the selected codeword exists in the old pool, it also reports that W's
errors from the old evaluation stream itself. That within-old-stream comparison
is particularly useful because MC samples and the Z pool are identical.
Outside-pool comparisons use new evaluation results and are descriptive, with
both sample counts recorded. No error-based codeword selection is performed.

If errors decrease, that supports sensitivity to precoder selection; it does
not by itself identify the missing fourth-moment term or prove the Gaussian-hop
model exact. A single pilot/reference Z also has selection variability; the
stored pilot seed offset permits a separate sensitivity experiment later.

## Validation

The new selector integration tests use a rank-one synthetic pilot channel with
an exact known codeword, verify the existing selector returns it even when it
is outside the baseline pool, and verify that Z is preserved. They also reject
invalid references and zero pilots. Torch was unavailable in this development
environment, so these three tests were skipped rather than reported as passed.
The five NumPy algebra tests of the unchanged analytic calculation passed.
Notebook JSON and all code cell syntax were validated. No 4000-bank MC run was
executed during development.
