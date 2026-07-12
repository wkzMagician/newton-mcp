# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Action-reaction records emitted by coupling solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CouplingExchange:
    """One impulse exchange across a multiphysics interface.

    Attributes:
        pair_type: Coupling kind, such as ``rigid-fluid``.
        object_a: First scene object identifier.
        object_b: Second scene object identifier.
        linear_impulse_a: Linear impulse applied to A [N s].
        linear_impulse_b: Linear impulse applied to B [N s].
        angular_impulse_a: Angular impulse applied to A [N m s].
        angular_impulse_b: Angular impulse applied to B [N m s].
        interface_residual: Dimensionless interface velocity residual.
        penetration: Maximum interface penetration [m].
    """

    pair_type: str
    object_a: str
    object_b: str
    linear_impulse_a: np.ndarray
    linear_impulse_b: np.ndarray
    angular_impulse_a: np.ndarray
    angular_impulse_b: np.ndarray
    interface_residual: float
    penetration: float
    contact_point: np.ndarray | None = None
    normal: np.ndarray | None = None
    barycentric_weights: np.ndarray | None = None
    normal_impulse: float = 0.0
    tangential_impulse: float = 0.0
    exchange_energy_error: float = 0.0

    @property
    def impulse_balance_error(self) -> float:
        """Return normalized action-reaction linear impulse error."""
        numerator = float(np.linalg.norm(self.linear_impulse_a + self.linear_impulse_b))
        denominator = float(
            np.linalg.norm(self.linear_impulse_a) + np.linalg.norm(self.linear_impulse_b) + 1.0e-12
        )
        return numerator / denominator

    @property
    def angular_impulse_balance_error(self) -> float:
        """Return normalized action-reaction angular impulse error."""
        numerator = float(np.linalg.norm(self.angular_impulse_a + self.angular_impulse_b))
        denominator = float(
            np.linalg.norm(self.angular_impulse_a)
            + np.linalg.norm(self.angular_impulse_b)
            + 1.0e-12
        )
        return numerator / denominator

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "pair_type": self.pair_type,
            "object_a": self.object_a,
            "object_b": self.object_b,
            "linear_impulse_a": self.linear_impulse_a.tolist(),
            "linear_impulse_b": self.linear_impulse_b.tolist(),
            "angular_impulse_a": self.angular_impulse_a.tolist(),
            "angular_impulse_b": self.angular_impulse_b.tolist(),
            "interface_residual": self.interface_residual,
            "penetration": self.penetration,
            "impulse_balance_error": self.impulse_balance_error,
            "angular_impulse_balance_error": self.angular_impulse_balance_error,
            "contact_point": self.contact_point.tolist() if self.contact_point is not None else None,
            "normal": self.normal.tolist() if self.normal is not None else None,
            "barycentric_weights": (
                self.barycentric_weights.tolist() if self.barycentric_weights is not None else None
            ),
            "normal_impulse": self.normal_impulse,
            "tangential_impulse": self.tangential_impulse,
            "exchange_energy_error": self.exchange_energy_error,
        }
