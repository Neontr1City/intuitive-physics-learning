"""Analytic events. SI units; hidden parameters stay outside model features."""

from dataclasses import dataclass

import numpy as np

G = 9.81
MU = (0.1, 0.5)
E = (0.2, 0.9)
PREFIXES = (0, 1, 2, 4, 8)
TASKS = ("slide", "collision_slide", "rebound")


def outcome(task, speed, threshold, mu, restitution):
    if task == 0:
        return speed**2 / (2 * mu * G) > threshold
    if task == 1:
        return (0.5 * (1 + restitution) * speed) ** 2 / (2 * mu * G) > threshold
    if task == 2:
        return restitution * speed > threshold
    raise ValueError(f"Unknown task: {task}")


def features(task, speed, threshold):
    identity = np.broadcast_to(np.eye(3)[task], (*speed.shape, 3))
    return np.concatenate(
        (identity, (speed / 3)[..., None], np.log1p(threshold)[..., None]), axis=-1
    ).astype(np.float32)


@dataclass
class Episodes:
    context: np.ndarray
    feedback: np.ndarray
    query: np.ndarray
    labels: np.ndarray
    mu: np.ndarray
    restitution: np.ndarray
    context_q: np.ndarray
    query_speed: np.ndarray
    query_threshold: np.ndarray


def generate(seed, n, per_task=24, condition="high"):
    """Shared context template permits matched cross-environment feedback shuffling.

    Calls with the same seed but different conditions share parameters, speeds,
    queries, and random draws. Query shape is [environment, task, event, feature].
    """
    if condition not in ("high", "low", "mixed") or n < 2 or per_task < 1:
        raise ValueError(
            "Need valid condition, at least two episodes and one query per task"
        )
    rng = np.random.default_rng(seed)
    mu = rng.uniform(*MU, n)
    restitution = rng.uniform(*E, n)
    speed = np.broadcast_to(rng.uniform(1, 3, 8), (n, 8)).copy()
    high = np.broadcast_to(rng.uniform(*MU, 8), (n, 8)).copy()
    low = np.broadcast_to(rng.choice([0.08, 0.6], 8), (n, 8)).copy()
    mixed = rng.random(n) < 0.5
    q = high if condition == "high" else low
    if condition == "mixed":
        q = np.where(mixed[:, None], high, low)
    distance = speed**2 / (2 * G * q)
    feedback = outcome(0, speed, distance, mu[:, None], restitution[:, None])
    qs = rng.uniform(1, 3, (n, 3, per_task))
    qt = rng.uniform(*MU, (n, 3, per_task))
    thresholds = qs**2 / (2 * G * qt)
    thresholds[:, 1] *= (0.5 * (1 + np.mean(E))) ** 2
    thresholds[:, 2] = qs[:, 2] * rng.uniform(*E, (n, per_task))
    query = np.stack([features(t, qs[:, t], thresholds[:, t]) for t in range(3)], 1)
    labels = np.stack(
        [
            outcome(t, qs[:, t], thresholds[:, t], mu[:, None], restitution[:, None])
            for t in range(3)
        ],
        1,
    )
    return Episodes(
        features(0, speed, distance),
        feedback.astype(np.float32),
        query,
        labels.astype(np.float32),
        mu,
        restitution,
        q,
        qs,
        thresholds,
    )


def oracle_predictions(data, quadrature=256):
    """Uniform mu posterior interval; independent e prior integrated by midpoints."""
    lo = np.full(len(data.mu), MU[0])
    hi = np.full(len(data.mu), MU[1])
    es = E[0] + (np.arange(quadrature) + 0.5) * (E[1] - E[0]) / quadrature
    predictions = []
    for k in range(9):
        if k in PREFIXES:
            p = np.empty_like(data.labels, dtype=np.float64)
            q = data.query_speed[:, 0] ** 2 / (2 * G * data.query_threshold[:, 0])
            p[:, 0] = np.clip((q - lo[:, None]) / (hi - lo)[:, None], 0, 1)
            # Chunk the quadrature to avoid a large environment x query x grid allocation.
            p[:, 1] = 0
            for e in es:
                q = (0.5 * (1 + e) * data.query_speed[:, 1]) ** 2 / (
                    2 * G * data.query_threshold[:, 1]
                )
                p[:, 1] += (
                    np.clip((q - lo[:, None]) / (hi - lo)[:, None], 0, 1) / quadrature
                )
            ratio = data.query_threshold[:, 2] / data.query_speed[:, 2]
            p[:, 2] = np.clip((E[1] - ratio) / (E[1] - E[0]), 0, 1)
            predictions.append(p)
        if k < 8:
            success = data.feedback[:, k].astype(bool)
            hi = np.where(success, np.minimum(hi, data.context_q[:, k]), hi)
            lo = np.where(~success, np.maximum(lo, data.context_q[:, k]), lo)
    return np.stack(predictions, 1)
