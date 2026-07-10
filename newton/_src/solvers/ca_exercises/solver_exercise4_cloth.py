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

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


@wp.func
def eval_spring_force(
    pos_a: wp.vec3,
    vel_a: wp.vec3,
    pos_b: wp.vec3,
    vel_b: wp.vec3,
    stiffness: float,
    damping: float,
    rest_length: float,
) -> wp.vec3:
    delta = pos_a - pos_b
    length = wp.length(delta)
    force = wp.vec3(0.0, 0.0, 0.0)

    if length > 1.0e-8:
        direction = delta / length
        force = stiffness * (rest_length - length) * direction  # 弹力
        force += damping * wp.dot(vel_b - vel_a, direction) * direction  # 阻尼力

    return force


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

    # Fixed particles at the top corners
    if idx == 0 or idx == grid_m - 1:
        particle_f[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    force = gravity * particle_mass

    pos = particle_q[idx]
    vel = particle_qd[idx]

    # Structural springs.
    if i > 0:
        n_idx = (i - 1) * grid_m + j
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_struct, damping, cell_size)
    if i + 1 < grid_n:
        n_idx = (i + 1) * grid_m + j
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_struct, damping, cell_size)
    if j > 0:
        n_idx = i * grid_m + j - 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_struct, damping, cell_size)
    if j + 1 < grid_m:
        n_idx = i * grid_m + j + 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_struct, damping, cell_size)

    # Shear springs.
    shear_rest = wp.sqrt(2.0) * cell_size
    if i > 0 and j > 0:
        n_idx = (i - 1) * grid_m + j - 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_shear, damping, shear_rest)
    if i > 0 and j + 1 < grid_m:
        n_idx = (i - 1) * grid_m + j + 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_shear, damping, shear_rest)
    if i + 1 < grid_n and j > 0:
        n_idx = (i + 1) * grid_m + j - 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_shear, damping, shear_rest)
    if i + 1 < grid_n and j + 1 < grid_m:
        n_idx = (i + 1) * grid_m + j + 1
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_shear, damping, shear_rest)

    # Bending springs.
    bend_rest = 2.0 * cell_size
    if i > 1:
        n_idx = (i - 2) * grid_m + j
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_bend, damping, bend_rest)
    if i + 2 < grid_n:
        n_idx = (i + 2) * grid_m + j
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_bend, damping, bend_rest)
    if j > 1:
        n_idx = i * grid_m + j - 2
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_bend, damping, bend_rest)
    if j + 2 < grid_m:
        n_idx = i * grid_m + j + 2
        force += eval_spring_force(pos, vel, particle_q[n_idx], particle_qd[n_idx], k_bend, damping, bend_rest)

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

    # Fixed particles at the top corners
    if idx == 0 or idx == grid_m - 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    acceleration = particle_f[idx] / particle_mass
    particle_q_out[idx] = particle_q[idx] + particle_qd[idx] * dt
    particle_qd_out[idx] = particle_qd[idx] + acceleration * dt


@wp.kernel
def semi_implicit_euler_integrate_kernel(
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

    if idx == 0 or idx == grid_m - 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    acceleration = particle_f[idx] / particle_mass
    velocity = particle_qd[idx] + acceleration * dt
    particle_qd_out[idx] = velocity
    particle_q_out[idx] = particle_q[idx] + velocity * dt


@wp.func
def spring_force_position_derivative(
    pos_a: wp.vec3,
    pos_b: wp.vec3,
    rest_length: float,
    stiffness: float,
) -> wp.mat33:
    pos_delta = pos_a - pos_b
    length = wp.length(pos_delta)
    if length <= 1.0e-8:
        return wp.mat33(0.0)

    direction = pos_delta / length
    outer = wp.outer(direction, direction)
    identity = wp.identity(n=3, dtype=wp.float32)

    # f_s = k * (r - l) * n
    # df_s / dx = k * (-n n^T + (r - l) / l * (I - n n^T))
    return stiffness * (-outer + ((rest_length - length) / length) * (identity - outer))


@wp.func
def spring_force_velocity_derivative(pos_a: wp.vec3, pos_b: wp.vec3, damping: float) -> wp.mat33:
    pos_delta = pos_a - pos_b
    length = wp.length(pos_delta)
    if length <= 1.0e-8:
        return wp.mat33(0.0)

    direction = pos_delta / length

    # f_d = c * ((v_b - v_a) dot n) * n
    # df_d / dv_a = -c * n n^T
    return -damping * wp.outer(direction, direction)


@wp.func
def _flat_vec3_component(values: wp.array(dtype=wp.vec3), flat_idx: int) -> float:
    vec = values[flat_idx // 3]
    component = flat_idx - (flat_idx // 3) * 3

    value = vec[2]
    if component == 0:
        value = vec[0]
    elif component == 1:
        value = vec[1]

    return value


@wp.func
def add_spring_jacobian_to_system(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    lhs: wp.array2d(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    particle_a: int,
    particle_b: int,
    rest_length: float,
    stiffness: float,
    damping: float,
    dt: float,
    system_size: int,
):
    dfdp = spring_force_position_derivative(particle_q[particle_a], particle_q[particle_b], rest_length, stiffness)
    dfdv = spring_force_velocity_derivative(particle_q[particle_a], particle_q[particle_b], damping)

    base_a = 3 * particle_a
    base_b = 3 * particle_b
    for row_component in range(3):
        row_a = base_a + row_component
        row_b = base_b + row_component

        for col_component in range(3):
            col_a = base_a + col_component
            col_b = base_b + col_component
            stiffness_block = dfdp[row_component, col_component]
            damping_block = dfdv[row_component, col_component]

            velocity_a = _flat_vec3_component(particle_qd, col_a)
            velocity_b = _flat_vec3_component(particle_qd, col_b)

            # Linearized backward Euler:
            #   (M - dt * D - dt^2 * K) dv = dt * (f + dt * K * v)
            #
            # For one spring, self blocks use K,D and cross blocks use -K,-D.
            lhs[row_a, col_a] += -dt * damping_block - dt * dt * stiffness_block
            lhs[row_a, col_b] += dt * damping_block + dt * dt * stiffness_block
            lhs[row_b, col_a] += dt * damping_block + dt * dt * stiffness_block
            lhs[row_b, col_b] += -dt * damping_block - dt * dt * stiffness_block

            rhs[row_a] += dt * dt * stiffness_block * (velocity_a - velocity_b)
            rhs[row_b] -= dt * dt * stiffness_block * (velocity_a - velocity_b)


@wp.kernel
def implicit_euler_build_system_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    lhs: wp.array2d(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    particle_mass: float,
    k_struct: float,
    k_bend: float,
    k_shear: float,
    damping: float,
    cell_size: float,
    dt: float,
    grid_n: int,
    grid_m: int,
    system_size: int,
):
    _ = wp.tid()
    particle_count = grid_n * grid_m

    for row in range(system_size):
        rhs[row] = 0.0
        for col in range(system_size):
            lhs[row, col] = 0.0
        lhs[row, row] = particle_mass

    for particle in range(particle_count):
        base = 3 * particle
        rhs[base + 0] = dt * particle_f[particle][0]
        rhs[base + 1] = dt * particle_f[particle][1]
        rhs[base + 2] = dt * particle_f[particle][2]

    for i in range(grid_n):
        for j in range(grid_m):
            idx = i * grid_m + j

            if i + 1 < grid_n:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    (i + 1) * grid_m + j,
                    cell_size,
                    k_struct,
                    damping,
                    dt,
                    system_size,
                )
            if j + 1 < grid_m:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    i * grid_m + j + 1,
                    cell_size,
                    k_struct,
                    damping,
                    dt,
                    system_size,
                )

            shear_rest = wp.sqrt(2.0) * cell_size
            if i + 1 < grid_n and j + 1 < grid_m:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    (i + 1) * grid_m + j + 1,
                    shear_rest,
                    k_shear,
                    damping,
                    dt,
                    system_size,
                )
            if i + 1 < grid_n and j > 0:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    (i + 1) * grid_m + j - 1,
                    shear_rest,
                    k_shear,
                    damping,
                    dt,
                    system_size,
                )

            bend_rest = 2.0 * cell_size
            if i + 2 < grid_n:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    (i + 2) * grid_m + j,
                    bend_rest,
                    k_bend,
                    damping,
                    dt,
                    system_size,
                )
            if j + 2 < grid_m:
                add_spring_jacobian_to_system(
                    particle_q,
                    particle_qd,
                    lhs,
                    rhs,
                    idx,
                    i * grid_m + j + 2,
                    bend_rest,
                    k_bend,
                    damping,
                    dt,
                    system_size,
                )

    for fixed_particle in range(2):
        fixed_idx = fixed_particle * (grid_m - 1)
        base = 3 * fixed_idx
        for component in range(3):
            row = base + component
            rhs[row] = 0.0
            for col in range(system_size):
                lhs[row, col] = 0.0
                lhs[col, row] = 0.0
            lhs[row, row] = 1.0


@wp.kernel
def implicit_euler_apply_solution_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
    delta_v: wp.array(dtype=wp.float32),
    dt: float,
    grid_m: int,
):
    idx = wp.tid()

    if idx == 0 or idx == grid_m - 1:
        particle_q_out[idx] = particle_q[idx]
        particle_qd_out[idx] = wp.vec3(0.0, 0.0, 0.0)
        return

    base = 3 * idx
    velocity = particle_qd[idx] + wp.vec3(delta_v[base], delta_v[base + 1], delta_v[base + 2])
    particle_qd_out[idx] = velocity
    particle_q_out[idx] = particle_q[idx] + dt * velocity


def cg_solve(
    A: wp.array,
    b: wp.array,
    x: wp.array,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> int:
    """Solve the linear system Ax = b using the Conjugate Gradient method."""
    device = A.device

    A_cpu = A.numpy()
    b_cpu = b.numpy()
    x_cpu = x.numpy().copy()

    r = b_cpu - A_cpu @ x_cpu
    p = r.copy()
    rs_old = r @ r

    for i in range(max_iters):
        Ap = A_cpu @ p
        pAp = p @ Ap
        if abs(pAp) < 1.0e-30:
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
            device=self.model.device,
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
            device=self.model.device,
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
            device=self.model.device,
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
            device=self.model.device,
        )

    def _step_implicit_euler(self, state_in: State, state_out: State, dt: float) -> None:
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
            device=self.model.device,
        )

        particle_mass = float(self.params.get("particle_mass", 1.0))
        system_size = 3 * self.particle_count
        lhs_wp = wp.zeros((system_size, system_size), dtype=wp.float32, device=self.model.device)
        rhs_wp = wp.zeros(system_size, dtype=wp.float32, device=self.model.device)
        delta_v_wp = wp.zeros(system_size, dtype=wp.float32, device=self.model.device)

        wp.launch(
            implicit_euler_build_system_kernel,
            dim=1,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                self.particle_f,
                lhs_wp,
                rhs_wp,
                particle_mass,
                self.params.get("k_struct", 1.0),
                self.params.get("k_bend", 1.0),
                self.params.get("k_shear", 1.0),
                self.params.get("damping", 1.0),
                self.params.get("cell_size", 1.0),
                dt,
                self.grid_n,
                self.grid_m,
                system_size,
            ],
            device=self.model.device,
        )

        cg_solve(lhs_wp, rhs_wp, delta_v_wp, max_iters=system_size, tol=1.0e-6)

        wp.launch(
            implicit_euler_apply_solution_kernel,
            dim=self.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
                delta_v_wp,
                dt,
                self.grid_m,
            ],
            device=self.model.device,
        )

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
