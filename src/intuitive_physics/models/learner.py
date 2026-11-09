"""Common evidence encoding, readout, and predict-then-update protocol."""

import torch
from torch import nn

from ..data import PREFIXES
from .euler import EulerUpdate
from .gru import GRUUpdate
from .node import NODEUpdate


class Learner(nn.Module):
    def __init__(self, kind="gru", state_dim=32, *, node_substeps=1, euler_substeps=4):
        super().__init__()
        if kind not in ("gru", "node", "euler"):
            raise ValueError(kind)
        self.kind = kind
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, state_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(state_dim + 5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        if kind == "gru":
            self.update = GRUUpdate(state_dim)
        elif kind == "node":
            self.update = NODEUpdate(state_dim, node_substeps)
        else:
            self.update = EulerUpdate(state_dim, euler_substeps)

    @property
    def substeps(self):
        return self.update.substeps

    @substeps.setter
    def substeps(self, value):
        if type(value) is not int or value < 1:
            raise ValueError("substeps must be a positive integer")
        self.update.substeps = value

    def predict(self, z, x):
        expanded = z[:, None].expand(-1, x.shape[1], -1)
        return self.readout(torch.cat((expanded, x), -1)).squeeze(-1)

    def absorb(self, z, x, y):
        p = self.predict(z, x[:, None]).sigmoid()
        # Backpropagate through the error feature during offline training.
        u = self.encoder(torch.cat((x, y[:, None], y[:, None] - p), -1))
        return self.update(z, u)

    def forward(self, context, feedback, query, frozen=False):
        z = context.new_zeros((context.shape[0], self.state_dim))
        logits = []
        for k in range(9):
            if k in PREFIXES:
                logits.append(self.predict(z, query))
            if k < 8 and not frozen:
                z = self.absorb(z, context[:, k], feedback[:, k])
        return torch.stack(logits, 1)
