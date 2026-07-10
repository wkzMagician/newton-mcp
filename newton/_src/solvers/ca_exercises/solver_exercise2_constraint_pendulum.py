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
    dt: float,
    ks: float,
    kd: float,
):
    if wp.tid() != 0:
        return

    x1 = body_q_in[0]
    x2 = body_q_in[1]
    qd1 = body_qd_in[0]
    qd2 = body_qd_in[1]

    p1 = x1.p
    p2 = x2.p
    v1 = wp.vec3(qd1[0], qd1[1], qd1[2])
    v2 = wp.vec3(qd2[0], qd2[1], qd2[2])

    r1 = p1 - origin
    r12 = p1 - p2
    v12 = v1 - v2

    C = wp.vec2(
        0.5 * (wp.dot(r1, r1) - radius1 * radius1),
        0.5 * (wp.dot(r12, r12) - radius2 * radius2),
    )
    Cdot = wp.vec2(
        wp.dot(r1, v1),
        wp.dot(r12, v12),
    )

    qd = vec6(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])
    f = vec6(gravity[0], gravity[1], gravity[2], gravity[0], gravity[1], gravity[2])

    J = mat26(
        r1[0],  r1[1],  r1[2],  0.0,    0.0,    0.0,
        r12[0], r12[1], r12[2], -r12[0], -r12[1], -r12[2],
    )
    Jdot = mat26(
        v1[0],  v1[1],  v1[2],  0.0,    0.0,    0.0,
        v12[0], v12[1], v12[2], -v12[0], -v12[1], -v12[2],
    )

    A = J @ wp.transpose(J)
    rhs = -(Jdot @ qd + J @ f) - (kd * Cdot + ks * C) # feedback
    lagrange = ldlt(A, rhs)

    f_tilde = wp.transpose(J) @ lagrange
    f1 = wp.vec3(f_tilde[0], f_tilde[1], f_tilde[2])
    f2 = wp.vec3(f_tilde[3], f_tilde[4], f_tilde[5])

    a1 = gravity + f1
    a2 = gravity + f2
    v1_new = v1 + a1 * dt
    v2_new = v2 + a2 * dt
    p1_new = p1 + v1_new * dt
    p2_new = p2 + v2_new * dt

    body_q_out[0] = wp.transform(p=p1_new, q=x1.q)
    body_q_out[1] = wp.transform(p=p2_new, q=x2.q)
    body_qd_out[0] = wp.spatial_vector(v1_new, wp.vec3(qd1[3], qd1[4], qd1[5]))
    body_qd_out[1] = wp.spatial_vector(v2_new, wp.vec3(qd2[3], qd2[4], qd2[5]))


class SolverExercise2ConstraintPendulum(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.origin = wp.vec3(0.0, 0.0, 0.0)
        self.radius1 = 5.0
        self.radius2 = 5.0

        self.gravity = -9.81
        self.ks = 40.0
        self.kd = 10.0

    def reset(self):
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
                self.ks,
                self.kd,
            ],
        )
