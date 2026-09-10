#!/usr/bin/env python3
"""Run from the repository after installation; see docs/CORRELATION_MC.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from ris_env.environment import BankInput, build_deterministic_bank
from ris_env.correlation_validation import run_correlation_mc
from ris_env.ris_response import generate_ris_response_from_z


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    temporary = path.with_suffix(path.suffix+".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def ris_shape(n):
    if n < 2 or n % 2:
        raise ValueError("nRIS must be even and >= 2 (includes both polarizations)")
    x = int((n//2)**0.5)
    while (n//2) % x:
        x -= 1
    return x, n//2//x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nris", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--n-mc", type=int, default=32_768)
    p.add_argument("--mc-chunk", type=int, default=128)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    p.add_argument("--patterns", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--float32", action="store_true", help="default: double precision validation")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--bank-json", type=Path, help="JSON list of BankInput dictionaries instead of demo banks")
    p.add_argument("--output", type=Path, default=Path("outputs/correlation_mc"))
    p.add_argument("--resume", action="store_true", help="skip completed banks with an identical manifest")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    if args.n_mc < 2 or args.mc_chunk < 1 or args.patterns < 1 or args.cpu_threads < 1:
        p.error("n-mc >= 2; mc-chunk, patterns and cpu-threads >= 1 are required")
    torch.set_num_threads(args.cpu_threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    if args.bank_json:
        banks = [BankInput(**entry) for entry in json.loads(args.bank_json.read_text())]
    else:
        banks = []
        for n in args.nris:
            nx, ny = ris_shape(n)
            for suffix in ("LOS", "NLOS"):
                banks.append(BankInput(
                    f"Indoor-Office-{suffix}", f"Indoor-Office-{suffix}", 3.5e9,
                    (0, 0, 3), (-10, 2, 3), (8, 3, 1.5),
                    2, 1, 2, 1, nx, ny, (1, 1, 2)))
    if not banks:
        p.error("at least one bank is required")
    # Resume only a compatible experiment, including the implementation itself.
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted((repo/"src/ris_env").glob("*.py")) + [Path(__file__)]:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    config = {"banks": [asdict(b) for b in banks], "n_mc": args.n_mc,
              "mc_chunk": args.mc_chunk, "patterns": args.patterns, "seed": args.seed,
              "checkpoints": args.checkpoints, "device": device, "parity": not args.float32,
              "cpu_threads": args.cpu_threads, "torch_version": str(torch.__version__),
              "source_sha256": digest.hexdigest()}
    config = json.loads(json.dumps(config))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/"manifest.json"
    if manifest.exists():
        if not args.resume:
            p.error("output already has a run; use --resume or a new --output directory")
        if json.loads(manifest.read_text()) != config:
            p.error("resume manifest differs; use a new --output directory")
    save_json(manifest, config)
    all_rows, all_checks = [], []
    for i, bank in enumerate(banks):
        n = 2*bank.nRIS_x*bank.nRIS_y
        name = f"bank_{i:03d}_nris{n}"
        result_path = args.output/f"{name}.json"
        metadata = dict(bank=i, nris=n, scenario_br=bank.scenario_br, scenario_ru=bank.scenario_ru)
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text())
            print(f"[{i+1}/{len(banks)}] {name}: completed, skipping", flush=True)
        else:
            print(f"[{i+1}/{len(banks)}] {name} {bank.scenario_br}/{bank.scenario_ru}: preparing", flush=True)
            prepared = build_deterministic_bank(bank, device=device, parity=not args.float32)
            rng = np.random.default_rng(args.seed+i)
            z = rng.integers(0, 2, (args.patterns, n), dtype=np.int8)
            z[0] = 0
            if args.patterns > 1:
                z[1] = 1
            if args.patterns > 2:
                z[2] = np.arange(n) % 2
            gamma = generate_ris_response_from_z(z, device=device, parity=not args.float32)["gamma"]
            partial = []
            def progress(rows):
                partial.extend({**metadata, **row} for row in rows)
                save_csv(args.output/f"{name}_progress.csv", partial)
                med = np.median([r["variance_rel_error"] for r in rows])*100
                print(f"  N={rows[0]['n_mc']:>7,}/{args.n_mc:,} | channel variance median error={med:.3f}% | {rows[0]['seconds']:.1f}s", flush=True)
            result = run_correlation_mc(
                prepared, gamma, n_mc=args.n_mc, mc_chunk=args.mc_chunk,
                checkpoints=args.checkpoints, seed=args.seed+1000*i, progress=progress)
            result["bank_input"] = asdict(bank)
            result["z"] = z.tolist()
            result["w_real"] = prepared["W"].real.cpu().tolist()
            result["w_imag"] = prepared["W"].imag.cpu().tolist()
            save_json(result_path, result)
        all_rows.extend({**metadata, **row} for row in result["rows"])
        all_checks.append({**metadata, **result["checks"], **result["timings"]})
        save_csv(args.output/"convergence.csv", all_rows)
        save_csv(args.output/"algebra_checks.csv", all_checks)
    final = pd.DataFrame(all_rows)
    final = final[final.n_mc == args.n_mc].copy()
    final.to_csv(args.output/"final_comparison.csv", index=False)
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.DataFrame(all_rows)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for bank_id, rows in df.groupby("bank"):
            med = rows.groupby("n_mc").variance_rel_error.median()*100
            ax[0].plot(med.index, med.values, "o-", label=f"bank {bank_id}, Nris={rows.nris.iloc[0]}")
        ax[0].set(xscale="log", yscale="log", xlabel="Monte Carlo samples", ylabel="Median channel variance error (%)")
        ax[0].legend(fontsize=8)
        powers = final.drop_duplicates(["bank", "candidate"])
        ax[1].scatter(powers.mc_power_variance, powers.wick_power_variance)
        lo = min(powers.mc_power_variance.min(), powers.wick_power_variance.min())
        hi = max(powers.mc_power_variance.max(), powers.wick_power_variance.max())
        ax[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax[1].set(xlabel="Native MC power variance", ylabel="Wick power variance", title="Power variance: separate diagnostic")
        for a in ax:
            a.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output/"comparison.png", dpi=160)
        plt.close(fig)
    # Algebra is a correctness gate. Native MC residuals are diagnostics because
    # supplied angular rho is approximate; do not hide them behind a PASS label.
    tol = 3e-4 if args.float32 else 1e-10
    bad = []
    for checks in all_checks:
        for key in ("four_sum_mean_rel_error", "four_sum_variance_rel_error",
                    "production_mean_rel_error", "production_UBR_rel_error", "sample_composition_rel_error"):
            if key in checks and (not np.isfinite(checks[key]) or checks[key] > tol):
                bad.append((checks["bank"], key, checks[key]))
    print(f"Saved {args.output.resolve()}", flush=True)
    if bad:
        print(f"Algebra checks failed: {bad}", file=sys.stderr)
        return 1
    print("Algebra checks passed. Inspect MC errors/SE and raw variance separately.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
