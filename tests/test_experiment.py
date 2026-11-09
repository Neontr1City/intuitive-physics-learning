import numpy as np
import pytest
import torch

from intuitive_physics.data import G, generate, oracle_predictions, outcome
from intuitive_physics.experiment import dataset, shuffle_feedback, tensors
from intuitive_physics.models.learner import Learner


def test_physics_boundaries_and_dependencies():
    stop = 4 / (2 * 0.2 * G)
    assert outcome(0, 2, stop - 1e-6, 0.2, 0.7)
    assert not outcome(0, 2, stop, 0.2, 0.7)
    assert not outcome(0, 2, stop, 0.4, 0.7)
    assert outcome(1, 2, 0.5, 0.2, 0.7)
    assert not outcome(1, 2, 0.5, 0.4, 0.7)
    assert outcome(2, 2, 1, 0.1, 0.7) == outcome(2, 2, 1, 0.5, 0.7)
    assert not outcome(2, 2, 1, 0.1, 0.2)


def test_paired_conditions_reproducibility_and_no_parameter_input():
    high, low = generate(1, 100), generate(1, 100, condition="low")
    for field in ("mu", "restitution", "query", "labels"):
        np.testing.assert_array_equal(getattr(high, field), getattr(low, field))
    np.testing.assert_array_equal(high.context, generate(1, 100).context)
    assert high.context.shape == (100, 8, 5)
    np.testing.assert_array_equal(
        high.context, np.broadcast_to(high.context[0], high.context.shape)
    )
    assert np.unique(high.mu).size == 100


def test_oracle_information_and_negative_control():
    high, low = generate(9, 1000), generate(9, 1000, condition="low")
    p, q = oracle_predictions(high), oracle_predictions(low)
    np.testing.assert_array_equal(p[:, 0, 2], p[:, -1, 2])
    np.testing.assert_array_equal(q[:, 0], q[:, -1])
    loss = ((p - high.labels[:, None]) ** 2).mean((0, 3))
    assert loss[0, 0] - loss[-1, 0] > 0.03
    assert loss[0, 1] - loss[-1, 1] > 0.01


def test_feedback_shuffle_matches_event_inputs():
    data = dataset(22, 128, 4, "high")
    x, y, _ = tensors(data)
    shuffled = shuffle_feedback(y)
    for start in (0, 64):
        assert torch.equal(shuffled[start], y[start + 63])
        assert torch.equal(x[start], x[start + 63])


@pytest.mark.parametrize("kind", ["gru", "node", "euler"])
def test_query_isolation_reset_freeze_and_gradients(kind):
    torch.manual_seed(1)
    model = Learner(kind)
    x, y, q = tensors(generate(1, 4, 3))
    a = model(x, y, q)
    b = model(x, y, q.flip(1))
    torch.testing.assert_close(a, b.flip(2))
    torch.testing.assert_close(a, model(x, y, q))
    frozen = model(x, y, q, frozen=True)
    torch.testing.assert_close(frozen[:, 0], frozen[:, -1])
    altered = model(x, 1 - y, q)
    torch.testing.assert_close(a[:, 0], altered[:, 0])
    assert not torch.allclose(a[:, -1], altered[:, -1])
    a[:, -1].square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_splits_differ():
    validation, test = dataset(20, 64, 4, "high"), dataset(30, 64, 4, "high")
    assert not np.array_equal(validation.mu, test.mu)
    assert not np.array_equal(validation.query, test.query)


def test_rk4_refinement_on_known_linear_flow():
    model = Learner("node", state_dim=2)
    model.update.field = lambda state, evidence: 0.5 * state
    z = torch.ones(3, 2)
    x, y = torch.zeros(3, 5), torch.zeros(3)
    coarse = model.absorb(z, x, y)
    model.substeps = 2
    refined = model.absorb(z, x, y)
    exact = z * np.exp(0.5)
    assert (refined - exact).abs().max() < (coarse - exact).abs().max()
    torch.testing.assert_close(refined, exact, atol=3e-5, rtol=0)
