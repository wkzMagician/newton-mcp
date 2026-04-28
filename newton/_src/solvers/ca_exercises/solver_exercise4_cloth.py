# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


@wp.kernel
def compute_particle_force_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_mass: float,
    k_struct: float,
    k_bend: float,
    k_shear: float,
    damping: float,
    cell_size: float,
    grid_n: int,
    grid_m: int,
    gravity: wp.vec3,
):
    i, j = wp.tid()
    idx = i * grid_m + j

    # Fixed particles at the top row
    if idx == 0 or idx == grid_m - 1:
        particle_f[idx] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    force = gravity * particle_mass

    # TODO: Add structural, bending, and shear spring forces here


    particle_f[idx] = force
    

@wp.kernel
def explicit_euler_integrate_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_mass: float,
    dt: float,
    grid_n: int,
    grid_m: int,
):
    idx = wp.tid()

    # Fixed particles at the top row
    if idx == 0 or idx == grid_m - 1:
        return

    # TODO: implement explicit Euler integration here



@wp.kernel
def semi_implicit_euler_integrate_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_mass: float,
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
):
    idx = wp.tid()

    if particle_flags[idx] == 0:
        return

    # TODO: implement semi-implicit Euler integration here



def cg_solve(
    A: wp.array,
    b: wp.array,
    x: wp.array,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> int:
    """Solve the linear system Ax = b using the Conjugate Gradient method."""
    device = A.device

    # Copy to CPU for computation
    A_cpu = A.numpy()
    b_cpu = b.numpy()
    x_cpu = x.numpy().copy()

    r = b_cpu - A_cpu @ x_cpu
    p = r.copy()
    rs_old = r @ r

    for i in range(max_iters):
        Ap = A_cpu @ p
        pAp = p @ Ap
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x_cpu = x_cpu + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new < tol * tol:
            wp.copy(x, wp.array(x_cpu, dtype=wp.float32, device=device))
            return i + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    wp.copy(x, wp.array(x_cpu, dtype=wp.float32, device=device))
    return max_iters


# TODO: Add appropriate kernels for implicit method here



class SolverExercise4Cloth(SolverBase):
    """Cloth solver with explicit, implicit, and semi-implicit Euler integration."""

    def __init__(
        self,
        model: Model,
        method: int = 0,
        params: dict | None = None,
        gravity: wp.vec3 | None = None,
    ):
        super().__init__(model)

        self.method = method
        self.params = params if params is not None else {}
        self.gravity = gravity if gravity is not None else wp.vec3(0.0, 0.0, -9.81)

        self.grid_n = self.params.get("grid_n", 16)
        self.grid_m = self.params.get("grid_m", 16)
        self.particle_count = self.grid_n * self.grid_m
        self.particle_f = wp.zeros(self.particle_count, dtype=wp.vec3, device=model.device)

    def _step_explicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        self.particle_f.fill_(0.0)

        wp.launch(
            compute_particle_force_kernel,
            dim=(self.grid_n, self.grid_m),
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.particle_f,
                self.params.get("particle_mass", 1.0),
                self.params.get("k_struct", 1.0),
                self.params.get("k_bend", 1.0),
                self.params.get("k_shear", 1.0),
                self.params.get("damping", 1.0),
                self.params.get("cell_size", 1.0),
                self.grid_n,
                self.grid_m,
                self.gravity,
            ],
        )

        wp.launch(
            explicit_euler_integrate_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
                self.particle_f,
                self.params.get("particle_mass", 1.0),
                dt,
                self.grid_n,
                self.grid_m,
            ],
        )

    def _step_semi_implicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        self.particle_f.fill_(0.0)

        wp.launch(
            compute_particle_force_kernel,
            dim=(self.grid_n, self.grid_m),
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.particle_f,
                self.params.get("particle_mass", 1.0),
                self.params.get("k_struct", 1.0),
                self.params.get("k_bend", 1.0),
                self.params.get("k_shear", 1.0),
                self.params.get("damping", 1.0),
                self.params.get("cell_size", 1.0),
                self.grid_n,
                self.grid_m,
                self.gravity,
            ],
        )

        wp.launch(
            semi_implicit_euler_integrate_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
                self.particle_f,
                self.params.get("particle_mass", 1.0),
                dt,
                self.grid_n,
                self.grid_m,
            ],
        )

    def _step_implicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
        # TODO: Implement implicit Euler method using Conjugate Gradient solver

        pass

    def reset(self) -> None:
        pass

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        if self.method == 0:
            self._step_explicit_euler(state_in, state_out, dt)
        elif self.method == 1:
            self._step_implicit_euler(state_in, state_out, dt)
        elif self.method == 2:
            self._step_semi_implicit_euler(state_in, state_out, dt)
        return None

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        # Cloth-ground interaction is handled directly in the kernels, not via contacts.
        _ = contacts
