"""Aggregate paired training seeds and context-template clusters across run directories."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import TASKS
from .experiment import plot_results, write_json


def uncertainty(values):
    """Resample training seeds and shared context groups; keep pairing intact."""
    grouped = values.reshape(values.shape[0], -1, 64).mean(-1)
    rng = np.random.default_rng(9201)
    estimates = []
    for _ in range(2000):
        seeds = rng.integers(0, len(grouped), len(grouped))
        groups = rng.integers(0, grouped.shape[1], grouped.shape[1])
        estimates.append(grouped[np.ix_(seeds, groups)].mean())
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "seed_means": values.mean(1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    merged = None
    scores = {}
    test_config = None
    for folder in args.runs:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        signature = (
            config.get("test_split", 30),
            config["episodes"],
            config["per_task"],
        )
        if test_config is not None and signature != test_config:
            raise ValueError("Cannot pair different test sets")
        test_config = signature
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if merged is None:
            merged = {"oracle": result["oracle"], "runs": {}}
        for key, run in result["runs"].items():
            if key in merged["runs"]:
                raise ValueError(f"Duplicate run: {key}")
            merged["runs"][key] = run
            scores[key] = {}
            for condition in ("high", "low"):
                with np.load(folder / f"{key}_{condition}_scores.npz") as arrays:
                    scores[key][condition] = dict(arrays)
    summary = {}
    lines = [
        "First analytic experiment",
        f"Test signature: {test_config}",
        "Brier gains: positive means improvement. CI resamples seeds and context groups.",
        "Intervals are exploratory, unadjusted for multiple comparisons.",
        "",
    ]
    for kind in sorted({key.split("_")[0] for key in scores}):
        keys = sorted(key for key in scores if key.startswith(kind + "_"))
        summary[kind] = {}
        lines.append(f"{kind}: {len(keys)} training seeds")
        for task, name in enumerate(TASKS):
            high = np.stack([scores[k]["high"]["correct"][:, :, task] for k in keys])
            low = np.stack([scores[k]["low"]["correct"][:, :, task] for k in keys])
            shuffled = np.stack(
                [scores[k]["high"]["shuffled"][:, -1, task] for k in keys]
            )
            effects = {
                "high_gain": uncertainty(high[:, :, 0] - high[:, :, -1]),
                "low_gain": uncertainty(low[:, :, 0] - low[:, :, -1]),
                "over_shuffled": uncertainty(shuffled - high[:, :, -1]),
                "high_over_low_gain": uncertainty(
                    (high[:, :, 0] - high[:, :, -1]) - (low[:, :, 0] - low[:, :, -1])
                ),
            }
            summary[kind][name] = effects
            lines.append(
                f"  {name}: Brier {high[:, :, 0].mean():.5f} -> {high[:, :, -1].mean():.5f}"
            )
            for effect, stats in effects.items():
                lines.append(
                    f"    {effect}: {stats['mean']:.5f} CI {stats['ci95']}; seeds {stats['seed_means']}"
                )
        runs = [merged["runs"][k] for k in keys]
        summary[kind]["cost"] = {
            "parameters": runs[0]["parameters"],
            "mean_train_seconds": float(np.mean([r["train_seconds"] for r in runs])),
            "best_steps": [r["best_step"] for r in runs],
        }
        lines.append(f"  Cost: {summary[kind]['cost']}")
    if "node" in summary and "euler" in summary:
        node_keys = sorted(k for k in scores if k.startswith("node_"))
        if {k.split("_")[1] for k in node_keys} != {
            k.split("_")[1] for k in scores if k.startswith("euler_")
        }:
            raise ValueError("Comparison requires matching training seeds")
        comparison = {}
        for task, name in enumerate(TASKS):
            differences = np.stack(
                [
                    scores[k.replace("node_", "euler_")]["high"]["correct"][:, -1, task]
                    - scores[k]["high"]["correct"][:, -1, task]
                    for k in node_keys
                ]
            )
            comparison[name] = uncertainty(differences)
        summary["node_advantage_over_euler"] = comparison
        lines.append(f"NODE advantage over Euler (positive favors NODE): {comparison}")
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "combined_results.json", merged)
    (args.output / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    plot_results(merged, args.output)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
