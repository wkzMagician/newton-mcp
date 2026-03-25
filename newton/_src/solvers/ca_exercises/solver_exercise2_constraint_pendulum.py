import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# NOTE: mat66/mat26/vec6 are complete and you can use them directly.
# DO NOT try to implement these classes.
#
# You can check the implementation of wp.types.matrix to learn how to use these classes
# Typical usage:
#   V = vec6(v0, v1, v2, v3, v4, v5)
#   M26 = mat26(m00, m01, m02, m03, m04, m05, m10, m11, m12, m13, m14, m15)
#   M66 = mat66(m00, m01, m02, m03, m04, m05, ..., m50, m51, m52, m53, m54, m55)

class vec6(wp.types.vector(length=6, dtype=float)):
    """6D vector with float (single-precision) components."""

class mat26(wp.types.matrix(shape=(2, 6), dtype=float)):
    """2x6 matrix with float (single-precision) components."""

class mat66(wp.types.matrix(shape=(6, 6), dtype=float)):
    """6x6 matrix with float (single-precision) components."""

@wp.func
def ldlt(A: wp.mat22, b: wp.vec2) -> wp.vec2:
    """Solve the linear system Ax = b for x."""
    eps = 1.0e-8

    a00 = A[0, 0]
    a01 = A[0, 1]
    a10 = A[1, 0]
    a11 = A[1, 1]

    d0 = a00
    if wp.abs(d0) < eps:
        d0 = eps

    l10 = a10 / d0
    d1 = a11 - l10 * a01
    if wp.abs(d1) < eps:
        d1 = eps

    y0 = b[0]
    y1 = b[1] - l10 * y0

    z0 = y0 / d0
    z1 = y1 / d1

    x1 = z1
    x0 = z0 - l10 * x1

    return wp.vec2(x0, x1)


@wp.kernel
def pendulum_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    origin: wp.vec3,
    radius1: float,
    radius2: float,
    gravity: wp.vec3,
    dt: float
):
    if wp.tid() != 0:
        return

    # TODO: Implement the pendulum constraint here
    # You can refer to the bead-on-wire kernel for some guidance
    # Hints:
    #   1. Body1 is at index 0, Body2 is at index 1. We do not have to calculate for
    #      each body, so only execute the kernel for tid() == 0
    #   2. You can set the mass of the bodies to 1.0 for simplicity
    #   3. The operator of matrix multiplication in Warp & Python is `@`, e.g. `C = A @ B`,
    #      as long as the shapes are compatible. Vector is columnar, e.g. vec3 is a 3x1 matrix
    #   4. You can use the classes and functions defined above, or implement your own 
    #      linear algebra utilities if you prefer

    x0 = body_q_in[0]
    v0 = body_qd_in[0]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    x1 = body_q_in[1]
    v1 = body_qd_in[1]
    p1 = x1.p
    lin_v1 = wp.vec3(v1[0], v1[1], v1[2])

    J = mat26(
        p0[0], p0[1], p0[2], 0.0, 0.0, 0.0,
        p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2], p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2],
    )
    J1 = mat26(
        v0[0], v0[1], v0[2], 0.0, 0.0, 0.0,
        v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2], v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2],
    )
    W = mat66(
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    )
    q1 = vec6(lin_v0[0], lin_v0[1], lin_v0[2], lin_v1[0], lin_v1[1], lin_v1[2])
    f = vec6(gravity[0], gravity[1], gravity[2], gravity[0], gravity[1], gravity[2])

    A = J @ W @ wp.transpose(J)
    b = -J1 @ q1 - J @ W @ f

    lambda_ = ldlt(A, b)
    F = wp.transpose(J) @ lambda_ + f
    a1 = F[0:3]
    a2 = F[3:6]

    lin_v0_new = lin_v0 + a1 * dt
    lin_v1_new = lin_v1 + a2 * dt
    p0_new = p0 + lin_v0_new * dt
    p1_new = p1 + lin_v1_new * dt

    body_q_out[0] = wp.transform(p=p0_new, q=x0.q)
    body_q_out[1] = wp.transform(p=p1_new, q=x1.q)
    body_qd_out[0] = wp.spatial_vector(lin_v0_new, wp.vec3(v0[3], v0[4], v0[5]))
    body_qd_out[1] = wp.spatial_vector(lin_v1_new, wp.vec3(v1[3], v1[4], v1[5]))


class SolverExercise2ConstraintPendulum(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.origin = wp.vec3(0.0, 0.0, 0.0)
        self.radius1 = 5.0
        self.radius2 = 5.0

        self.gravity = -9.81

    def reset(self):
        # TODO: reset any additional attributes you added in the constructor here
        
        pass

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        wp.launch(
            pendulum_kernel, 
            dim=state_in.body_q.shape[0], 
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.origin,
                self.radius1,
                self.radius2,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )
