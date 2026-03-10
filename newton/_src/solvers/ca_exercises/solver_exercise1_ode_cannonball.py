import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

"""
Notes: Newton uses warp as its base language, which is similar to python,
but with some additional features. You can utilize the GPU by writing simple
kernel functions instead explicitly call the cuda or other gpu apis.

In this exercise, Actually there is only 1 body in the scene, but we also
use kernel functions to manage the states, which helps you get familiar with 
the syntax of warp and how the kernel functions work.
"""

@wp.kernel
def analytic_kernel(
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    initial_poses: wp.array(dtype=wp.transform),
    initial_velocities: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    time: float
):
    i = wp.tid()

    # We do not consider angular motion here

    # Initial position and velocity
    x0 = initial_poses[i]
    v0 = initial_velocities[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # New position and velocity
    p = p0 + lin_v0 * time + 0.5 * gravity * (time * time)
    v = lin_v0 + gravity * time

    # Update the output state
    body_q_out[i] = wp.transform(p=p, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v[0], v[1], v[2], v0[3], v0[4], v0[5])

@wp.kernel
def explicit_euler_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    dt: float
):
    # TODO: Implement the explicit Euler method here
    # You can refer to the analytic solution for some guidance

    pass

# TODO: Implement the last 3 kernels here


class SolverExercise1ODECannonBall(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.method = 0 # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

        self.gravity = -9.81

        # For the analytic solution, we will need to store the initial
        # poses and velocities of the bodies, as well as a time variable
        # to track the simulation time.
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

    def reset(self):
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

    def analytic(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            analytic_kernel,
            dim=self.model.body_count,
            inputs=[
                state_out.body_q,
                state_out.body_qd,
                self.initial_poses,
                self.initial_velocities,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.time,
            ],
        )
        self.time += dt

    def explicit_euler(self, state_in: State, state_out: State, dt: float):
        wp.launch(
            explicit_euler_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )

    def semi_implicit_euler(self, state_in: State, state_out: State, dt: float):
        # TODO: launch the semi-implicit Euler kernel here

        pass

    def mid_point(self, state_in: State, state_out: State, dt: float):
        # TODO: launch the mid-point kernel here

        pass

    def rk4(self, state_in: State, state_out: State, dt: float):
        # TODO: launch the RK4 kernel here

        pass

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        if self.method == 0:
            self.analytic(state_in, state_out, dt)
        elif self.method == 1:
            self.explicit_euler(state_in, state_out, dt)
        elif self.method == 2:
            self.semi_implicit_euler(state_in, state_out, dt)
        elif self.method == 3:
            self.mid_point(state_in, state_out, dt)
        elif self.method == 4:
            self.rk4(state_in, state_out, dt)
