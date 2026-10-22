"""Local: simulator public surface. Global: callers need only Simulation and StepResult."""

from .simulation import Simulation, StepResult  # Local: hide rule-phase implementation modules behind one API.

__all__ = ["Simulation", "StepResult"]  # Local: explicit supported imports reduce coupling.
