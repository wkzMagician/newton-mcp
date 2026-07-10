import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

@wp.kernel
def bead_on_wire_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    origin: wp.vec3,
    radius: float,
    gravity: wp.vec3,
    dt: float
):
    i = wp.tid()

    # Current position and velocity
    x = body_q_in[i]
    v = body_qd_in[i]
    p = x.p
    lin_v = wp.vec3(v[0], v[1], v[2])

    radial = p - origin
    lagrange = -(wp.dot(gravity, radial) + wp.dot(lin_v, lin_v)) / wp.dot(radial, radial)
    accel = gravity + lagrange * radial
    
    lin_v_new = lin_v + accel * dt
    p_new = p + lin_v_new * dt
    
    # position-based fix
    radial_new = p_new - origin
    radial_len = wp.length(radial_new)
    eps = 1e-6
    if radial_len > eps:
        # 先把位置投影回圆周
        n = radial_new / radial_len
        p_new = origin + n * radius

        # 再把速度投影到切向方向：去掉径向分量
        lin_v_new = lin_v_new - wp.dot(lin_v_new, n) * n

    # Update the output state
    body_q_out[i] = wp.transform(p=p_new, q=x.q)
    body_qd_out[i] = wp.spatial_vector(lin_v_new, wp.vec3(v[3], v[4], v[5]))

class SolverExercise2ConstraintWire(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.origin = wp.vec3(0.0, 0.0, 0.0)
        self.radius = 5.0

        self.gravity = -9.81

    def reset(self):
        # TODO: reset any additional attributes you added in the constructor here

    @override
    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> State | None:
        wp.launch(
            bead_on_wire_kernel, 
            dim=state_in.body_q.shape[0], 
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.origin,
                self.radius,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                dt,
            ],
        )
