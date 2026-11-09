"""Run the first controlled experiment: uv run python -m intuitive_physics.experiment."""

import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .config import load_config
from .data import PREFIXES, TASKS, Episodes, generate, oracle_predictions
from .models.learner import Learner


def dataset(split, n, per_task, condition):
    groups = [
        generate([split, i], min(64, n - i), per_task, condition)
        for i in range(0, n, 64)
    ]
    return Episodes(
        **{
            f.name: np.concatenate([getattr(d, f.name) for d in groups])
            for f in fields(Episodes)
        }
    )


def tensors(data):
    return (
        torch.from_numpy(data.context),
        torch.from_numpy(data.feedback),
        torch.from_numpy(data.query.reshape(len(data.mu), -1, 5)),
    )


def shuffle_feedback(y):
    # Each 64-environment group has exactly the same event template.
    return torch.cat([part.roll(1, 0) for part in y.split(64)])


@torch.no_grad()
def predict(model, data, mode="correct"):
    model.eval()
    x, y, q = tensors(data)
    if mode == "shuffled":
        y = shuffle_feedback(y)
    result = [
        model(x[i : i + 64], y[i : i + 64], q[i : i + 64], frozen=mode == "frozen")
        .sigmoid()
        .numpy()
        for i in range(0, len(y), 64)
    ]
    return np.concatenate(result).reshape(len(y), len(PREFIXES), 3, -1)


def episode_scores(p, labels):
    return ((p - labels[:, None]) ** 2).mean(-1)


def interval(values):
    """Bootstrap shared 64-episode templates, retaining shuffle dependencies."""
    rng = np.random.default_rng(4401)
    groups = values.reshape(-1, 64).mean(1)
    estimates = np.array(
        [rng.choice(groups, len(groups), replace=True).mean() for _ in range(1000)]
    )
    return [float(values.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def train(kind, seed, args, validation, out):
    torch.manual_seed(seed)
    model_options = {
        "state_dim": args.state_dim,
        "node_substeps": args.node_substeps,
        "euler_substeps": args.euler_substeps,
    }
    model = Learner(kind, **model_options)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    best_state = None
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        batch = generate([10, seed, step], args.batch_size, args.per_task, "mixed")
        logits = model(*tensors(batch))
        target = torch.from_numpy(batch.labels.reshape(args.batch_size, -1))[
            :, None
        ].expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {kind}, seed {seed}, step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        if step % args.validation_every == 0 or step == args.steps:
            scores = []
            for data in validation.values():
                p = np.clip(predict(model, data), 1e-7, 1 - 1e-7)
                y = data.labels[:, None]
                scores.append(float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()))
            validation_loss = float(np.mean(scores))
            history.append(
                {
                    "step": step,
                    "train_bce": loss.item(),
                    "validation_bce": validation_loss,
                }
            )
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{kind} seed={seed} step={step} train={loss.item():.4f} val={validation_loss:.4f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    torch.save(
        {
            "kind": kind,
            "model_config": model_options,
            "state_dict": best_state,
            "seed": seed,
            "best_step": best_step,
        },
        out / f"{kind}_{seed}.pt",
    )
    info = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": seconds,
        "best_step": best_step,
        "validation_bce": best_loss,
        "update_evaluations_per_event": model.update.evaluations_per_event,
        "history": history,
    }
    return model, info


def evaluate(model, data, stem, out):
    arrays, summary = {}, {}
    for mode in ("correct", "frozen", "shuffled"):
        p = predict(model, data, mode)
        arrays[mode] = episode_scores(p, data.labels)
        summary[mode] = {
            "brier": arrays[mode].mean(0).tolist(),
            "accuracy": ((p >= 0.5) == data.labels[:, None]).mean((0, 3)).tolist(),
        }
    for task, name in enumerate(TASKS):
        summary[name] = {
            "gain_mean_ci": interval(
                arrays["correct"][:, 0, task] - arrays["correct"][:, -1, task]
            ),
            "over_shuffled_mean_ci": interval(
                arrays["shuffled"][:, -1, task] - arrays["correct"][:, -1, task]
            ),
        }
    if model.kind == "node":
        original_substeps = model.substeps
        try:
            model.substeps = original_substeps * 2
            refined = predict(model, data)
        finally:
            model.substeps = original_substeps
        original = predict(model, data)
        summary["solver_refinement"] = {
            "mean_absolute_probability_change": float(abs(refined - original).mean()),
            "max_absolute_probability_change": float(abs(refined - original).max()),
        }
    np.savez_compressed(out / f"{stem}_scores.npz", **arrays)
    return summary


def plot_results(report, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for row, condition in enumerate(("high", "low")):
        for task, name in enumerate(TASKS):
            ax = axes[row, task]
            oracle = np.array(report["oracle"][condition]["brier"])
            ax.plot(
                PREFIXES, oracle[:, task], "k--", label="physics-informed reference"
            )
            for kind in sorted({run.split("_")[0] for run in report["runs"]}):
                curves = [
                    np.array(run[condition]["correct"]["brier"])[:, task]
                    for key, run in report["runs"].items()
                    if key.startswith(kind + "_")
                ]
                curves = np.array(curves)
                ax.plot(PREFIXES, curves.mean(0), marker="o", label=kind)
                ax.fill_between(PREFIXES, curves.min(0), curves.max(0), alpha=0.12)
            ax.set(
                title=f"{condition} information: {name}",
                xlabel="Observations",
                ylabel="Brier score",
            )
            ax.legend(fontsize=8)
            ax.set_ylim(0, 0.25)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without running",
    )
    parser.add_argument("--models", nargs="+", choices=["gru", "node", "euler"])
    parser.add_argument("--seeds", nargs="+", type=int)
    for name in (
        "steps",
        "episodes",
        "per-task",
        "threads",
        "test-split",
        "state-dim",
        "batch-size",
        "node-substeps",
        "euler-substeps",
        "validation-every",
        "validation-episodes",
    ):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gradient-clip", type=float)
    supplied = vars(parser.parse_args(argv))
    path = supplied.pop("config", None)
    output = supplied.pop("output")
    dry_run = supplied.pop("dry_run", False)
    try:
        resolved = load_config(path, supplied)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return argparse.Namespace(
        **asdict(resolved),
        output=output,
        config_path=str(path.resolve()) if path else None,
        dry_run=dry_run,
    )


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args) | {"output": str(args.output)}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[2]
    snapshot = args.output / "source"
    snapshot.mkdir()
    hashes = {}
    package = Path(__file__).parent
    for source in [*package.rglob("*.py"), root / "pyproject.toml", root / "uv.lock"]:
        relative = (
            source.relative_to(package)
            if source.is_relative_to(package)
            else Path(source.name)
        )
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.config_path:
        shutil.copy2(args.config_path, args.output / "requested_config.toml")
    config = vars(args) | {
        "output": str(args.output),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": hashes,
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ),
    }
    write_json(args.output / "config.json", config)
    validation = {
        c: dataset(20, args.validation_episodes, args.per_task, c)
        for c in ("high", "low")
    }
    test = {
        c: dataset(args.test_split, args.episodes, args.per_task, c)
        for c in ("high", "low")
    }
    report = {"oracle": {}, "runs": {}, "diagnostics": {}}
    for condition, data in test.items():
        np.savez_compressed(
            args.output / f"test_{condition}.npz",
            **{f.name: getattr(data, f.name) for f in fields(Episodes)},
        )
        p = oracle_predictions(data)
        scores = episode_scores(p, data.labels)
        report["oracle"][condition] = {
            "brier": scores.mean(0).tolist(),
            "gain": {
                name: interval(scores[:, 0, t] - scores[:, -1, t])
                for t, name in enumerate(TASKS)
            },
        }
        report["diagnostics"][condition] = {
            "positive_fraction": data.labels.mean((0, 2)).tolist(),
            "mu_e_correlation": float(np.corrcoef(data.mu, data.restitution)[0, 1]),
        }
    write_json(args.output / "results.json", report)
    for kind in args.models:
        for seed in args.seeds:
            model, info = train(kind, seed, args, validation, args.output)
            for condition, data in test.items():
                info[condition] = evaluate(
                    model, data, f"{kind}_{seed}_{condition}", args.output
                )
            report["runs"][f"{kind}_{seed}"] = info
            write_json(args.output / "results.json", report)
            plot_results(report, args.output)
            print(
                json.dumps({f"{kind}_{seed}": {t: info["high"][t] for t in TASKS}}),
                flush=True,
            )


if __name__ == "__main__":
    main()
