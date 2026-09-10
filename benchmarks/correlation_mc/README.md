# CPU validation snapshot

30 tests passed. Four native channel banks, 8,192 samples per bank, fixed Type-I W and four binary RIS patterns. Double precision; PyTorch 2.14.0+cpu.

| nRIS | Scenario | Median channel variance error | Max channel variance error |
|---|---|---:|---:|
| 8 | Indoor-Office-LOS | 1.094% | 2.882% |
| 8 | Indoor-Office-NLOS | 1.860% | 4.295% |
| 32 | Indoor-Office-LOS | 0.330% | 0.946% |
| 32 | Indoor-Office-NLOS | 1.758% | 4.269% |

Maximum four-sum relative error: 6.214e-16.
Maximum production channel variance relative error: 7.692e-16.

These are small-array CPU results, not validation of the 128/512 CUDA experiment. The Gaussian scalar trace identity has an independent MC unit test; native MIMO power variance remains a separate Wick comparison.

Configuration, source fingerprint, command and results are recorded in `cpu_summary.json`. CSV files retain per-candidate/per-receiver data and convergence checkpoints. See [the validation guide](../../docs/CORRELATION_MC.md) for interpretation.
