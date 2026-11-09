from dataclasses import asdict
from pathlib import Path

import pytest

from intuitive_physics.config import RunConfig, load_config
from intuitive_physics.experiment import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_original_defaults_and_explicit_overrides():
    assert asdict(load_config()) == asdict(RunConfig())
    args = parse_args(
        [
            "--config",
            str(ROOT / "configs/first_version.toml"),
            "--output",
            "unused",
            "--models",
            "node",
            "--steps",
            "7",
            "--node-substeps",
            "2",
        ]
    )
    assert args.models == ["node"]
    assert args.steps == 7
    assert args.node_substeps == 2
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.test_split == 31
    assert args.batch_size == 64


def test_typo_rejected_instead_of_silently_using_default(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("[train]\nlearning_rae = 0.1\n")
    with pytest.raises(ValueError, match="learning_rae"):
        load_config(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"test_split": 20},
        {"episodes": 65},
        {"validation_episodes": 1},
        {"node_substeps": 0},
        {"lr": float("nan")},
        {"steps": True},
        {"models": ["gru", "gru"]},
        {"seeds": [0, 0]},
    ],
)
def test_invalid_protocol_or_overwriting_runs_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(overrides=overrides)


def test_smoke_and_comparison_select_all_models():
    smoke = load_config(ROOT / "configs/smoke.toml")
    comparison = load_config(ROOT / "configs/first_version.toml")
    assert smoke.models == comparison.models == ["gru", "node", "euler"]
    assert smoke.steps == 2
    assert comparison.steps == 3000
