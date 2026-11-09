"""Shared vector-field architecture for the matched RK4/Euler comparison."""

import torch
from torch import nn


class VectorField(nn.Sequential):
    def __init__(self, state_dim, substeps):
        super().__init__(
            nn.Linear(2 * state_dim, 64), nn.Tanh(), nn.Linear(64, state_dim), nn.Tanh()
        )
        self.substeps = substeps

    def field(self, state, evidence):
        return super().forward(torch.cat((state, evidence), -1))
