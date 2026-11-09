"""Small TOML experiment configuration; no inheritance or sweep framework."""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    models: list[str] = field(default_factory=lambda: ["gru"])
    seeds: list[int] = field(default_factory=lambda: [0])
    threads: int = 2
    state_dim: int = 32
    node_substeps: int = 1
    euler_substeps: int = 4
    steps: int = 1000
    batch_size: int = 64
    lr: float = 0.001
    gradient_clip: float = 5.0
    validation_every: int = 100
    per_task: int = 24
    episodes: int = 1024
    validation_episodes: int = 256
    test_split: int = 30


SECTIONS = {
    "model": {"state_dim": "state_dim"},
    "node": {"substeps": "node_substeps"},
    "euler": {"substeps": "euler_substeps"},
    "train": {
        "steps": "steps",
        "batch_size": "batch_size",
        "learning_rate": "lr",
        "gradient_clip": "gradient_clip",
        "validation_every": "validation_every",
    },
    "data": {"queries_per_task": "per_task"},
    "eval": {
        "episodes": "episodes",
        "validation_episodes": "validation_episodes",
        "test_split": "test_split",
    },
}


def load_config(path=None, overrides=None):
    values = asdict(RunConfig())
    if path is not None:
        document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in document.items():
            if key in ("models", "seeds", "threads"):
                values[key] = value
            elif key in SECTIONS and isinstance(value, dict):
                for name, setting in value.items():
                    if name not in SECTIONS[key]:
                        raise ValueError(f"Unknown configuration key: {key}.{name}")
                    values[SECTIONS[key][name]] = setting
            else:
                raise ValueError(f"Unknown configuration key or invalid section: {key}")
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"Unknown override: {key}")
        values[key] = value
    config = RunConfig(**values)
    validate(config)
    return config


def validate(config):
    for name in (
        "threads",
        "state_dim",
        "node_substeps",
        "euler_substeps",
        "steps",
        "batch_size",
        "validation_every",
        "per_task",
        "episodes",
        "validation_episodes",
    ):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    for name in ("episodes", "validation_episodes"):
        if getattr(config, name) % 64:
            raise ValueError(
                f"{name} must be a multiple of 64 (matched context groups)"
            )
    for name in ("lr", "gradient_clip"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not isinstance(config.models, list)
        or not config.models
        or any(m not in ("gru", "node", "euler") for m in config.models)
    ):
        raise ValueError("models must be a nonempty list of gru, node, euler")
    if (
        not isinstance(config.seeds, list)
        or not config.seeds
        or any(type(s) is not int or s < 0 for s in config.seeds)
    ):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(config.models)) != len(config.models) or len(set(config.seeds)) != len(
        config.seeds
    ):
        raise ValueError("Duplicate models or seeds would overwrite checkpoints")
    if (
        type(config.test_split) is not int
        or config.test_split < 0
        or config.test_split in (10, 20)
    ):
        raise ValueError(
            "test_split must be nonnegative; 10/20 are reserved for train/validation"
        )
