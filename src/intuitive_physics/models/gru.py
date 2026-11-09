"""One gated recurrent state update per feedback event."""

from torch import nn


class GRUUpdate(nn.GRUCell):
    def __init__(self, state_dim):
        super().__init__(state_dim, state_dim)

    def forward(self, state, evidence):
        return super().forward(evidence, state)

    @property
    def evaluations_per_event(self):
        return 1
