"""Repeated Euler updates with the same vector-field architecture as NODE."""

from .field import VectorField


class EulerUpdate(VectorField):
    def __init__(self, state_dim, substeps=4):
        super().__init__(state_dim, substeps)

    def forward(self, state, evidence):
        h = 1 / self.substeps
        for _ in range(self.substeps):
            state = state + h * self.field(state, evidence)
        return state

    @property
    def evaluations_per_event(self):
        return self.substeps
