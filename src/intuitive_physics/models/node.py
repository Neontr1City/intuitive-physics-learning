"""Integrate the evidence-conditioned vector field over s in [0, 1] using RK4."""

from .field import VectorField


class NODEUpdate(VectorField):
    def __init__(self, state_dim, substeps=1):
        super().__init__(state_dim, substeps)

    def forward(self, state, evidence):
        h = 1 / self.substeps
        for _ in range(self.substeps):
            a = self.field(state, evidence)
            b = self.field(state + h * a / 2, evidence)
            c = self.field(state + h * b / 2, evidence)
            d = self.field(state + h * c, evidence)
            state = state + h * (a + 2 * b + 2 * c + d) / 6
        return state

    @property
    def evaluations_per_event(self):
        return 4 * self.substeps
