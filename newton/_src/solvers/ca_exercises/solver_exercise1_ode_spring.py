import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


@wp.func
def compute_spring_force(
    p: wp.vec3, v: wp.vec3, p0: wp.vec3, gravity: wp.vec3, mass: float, k: float, L: float, gamma: float
) -> wp.vec3:
    """
    计算弹簧力.

    f = -k(||p - p0|| - L) * n + mg - gamma * v
    where n = (p - p0) / ||p - p0||
    """
    # 锚点指向当前位置的方向
    diff = p - p0
    dist = wp.length(diff)

    # Spring force (-k * (dist - L) * n)
    f_spring = wp.vec3(0.0, 0.0, 0.0)
    if dist > 0.0001:
        n = diff / dist
        f_spring = -k * (dist - L) * n

    # 重力
    f_gravity = mass * gravity

    # 阻尼力
    f_damping = -gamma * v

    return f_spring + f_gravity + f_damping


@wp.kernel
def analytic_spring_kernel(
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    initial_poses: wp.array(dtype=wp.transform),
    initial_velocities: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    spring_rest_length: float,
    spring_stiffness: float,
    spring_damping: float,
    mass: float,
    time: float,
):
    """
    一维弹簧-质量-阻尼系统解析解.

    x = mg/k + e^(alphat) * (A*cos(betat) + B*sin(betat))
    where alpha = -gamma/(2m), beta = sqrt(4mk - gamma²)/(2m)
    A = -mg/k, B = -alpha*A/beta

    Real position: x_cube = -x(t) - L
    Real velocity: v_cube = -x'(t)
    """
    i = wp.tid()

    # 获取初始状态
    x0 = initial_poses[i]
    v0 = initial_velocities[i]
    wp.vec3(v0[0], v0[1], v0[2])

    # 系统沿z轴
    g = gravity[2]  # z方向重力

    # 计算参数
    mg_over_k = mass * g / spring_stiffness

    # alpha = -gamma/(2m)
    alpha = -spring_damping / (2.0 * mass)

    # beta = sqrt(4mk - gamma²)/(2m)
    discriminant = 4.0 * mass * spring_stiffness - spring_damping * spring_damping
    beta = wp.sqrt(discriminant) / (2.0 * mass)

    # A = -mg/k, B = -alpha*A/beta
    A = -mg_over_k
    B = -alpha * A / beta

    # x(t) = mg/k + e^(alphat) * (A*cos(betat) + B*sin(betat))
    exp_term = wp.exp(alpha * time)
    cos_term = wp.cos(beta * time)
    sin_term = wp.sin(beta * time)

    x_t = mg_over_k + exp_term * (A * cos_term + B * sin_term)

    # x'(t) = e^(alphat) * [(alpha*A + B*beta)*cos(betat) + (alpha*B - A*beta)*sin(betat)]
    x_dot_t = exp_term * ((alpha * A + B * beta) * cos_term + (alpha * B - A * beta) * sin_term)

    # Real position: x_cube = -x(t) + y_s - L (y_s = 0 is anchor position)
    p_z = -x_t - spring_rest_length

    # Real velocity: v_cube = -x'(t)
    v_z = -x_dot_t

    p_new = wp.vec3(0.0, 0.0, p_z)
    v_new = wp.vec3(0.0, 0.0, v_z)

    # 更新输出状态
    body_q_out[i] = wp.transform(p=p_new, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v0[3], v0[4], v0[5])


@wp.kernel
def explicit_euler_spring_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    spring_rest_length: float,
    spring_stiffness: float,
    spring_damping: float,
    mass: float,
    dt: float,
):
    """Explicit Euler integration for spring system / 弹簧系统显式欧拉积分."""
    i = wp.tid()

    # 获取当前状态
    x0 = body_q_in[i]
    v0 = body_qd_in[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # 锚点在原点
    p_anchor = wp.vec3(0.0, 0.0, 0.0)

    # 计算当前状态的力
    f = compute_spring_force(p0, lin_v0, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)

    # Explicit Euler: v_new = v_old + (f/m) * dt, p_new = p_old + v_old * dt
    a = f / mass
    v_new = lin_v0 + a * dt
    p_new = p0 + lin_v0 * dt

    # 更新输出状态
    body_q_out[i] = wp.transform(p=p_new, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v0[3], v0[4], v0[5])


@wp.kernel
def semi_implicit_euler_spring_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    spring_rest_length: float,
    spring_stiffness: float,
    spring_damping: float,
    mass: float,
    dt: float,
):
    """Semi-Implicit Euler integration for spring system / 弹簧系统半隐式欧拉积分."""
    i = wp.tid()

    # 获得当前状态
    x0 = body_q_in[i]
    v0 = body_qd_in[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # 锚点在原点
    p_anchor = wp.vec3(0.0, 0.0, 0.0)

    # 计算当前状态的力
    f = compute_spring_force(p0, lin_v0, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)

    # 半隐式欧拉:先更新速度,再用新速度更新位置
    a = f / mass
    v_new = lin_v0 + a * dt
    p_new = p0 + v_new * dt

    # 更新输出状态
    body_q_out[i] = wp.transform(p=p_new, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v0[3], v0[4], v0[5])


@wp.kernel
def mid_point_spring_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    spring_rest_length: float,
    spring_stiffness: float,
    spring_damping: float,
    mass: float,
    dt: float,
):
    """Mid-point integration for spring system / 弹簧系统中点法积分."""
    i = wp.tid()

    # 获得当前状态
    x0 = body_q_in[i]
    v0 = body_qd_in[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # 锚点在原点
    p_anchor = wp.vec3(0.0, 0.0, 0.0)

    # 步骤1:在t时刻评估导数
    f1 = compute_spring_force(p0, lin_v0, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)
    k1_v = f1 / mass
    k1_p = lin_v0

    # 步骤2:用k1计算中点导数
    v_mid = lin_v0 + k1_v * dt * 0.5
    p_mid = p0 + k1_p * dt * 0.5

    f2 = compute_spring_force(
        p_mid, v_mid, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping
    )
    k2_v = f2 / mass
    k2_p = v_mid

    # 用k2更新
    v_new = lin_v0 + k2_v * dt
    p_new = p0 + k2_p * dt

    # 更新输出状态
    body_q_out[i] = wp.transform(p=p_new, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v0[3], v0[4], v0[5])


@wp.kernel
def rk4_spring_kernel(
    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    spring_rest_length: float,
    spring_stiffness: float,
    spring_damping: float,
    mass: float,
    dt: float,
):
    """RK4 integration for spring system / 弹簧系统RK4积分."""
    i = wp.tid()

    # 获得当前状态
    x0 = body_q_in[i]
    v0 = body_qd_in[i]
    p0 = x0.p
    lin_v0 = wp.vec3(v0[0], v0[1], v0[2])

    # 锚点在原点
    p_anchor = wp.vec3(0.0, 0.0, 0.0)

    # k1: derivatives at t
    f1 = compute_spring_force(p0, lin_v0, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)
    k1_v = f1 / mass
    k1_p = lin_v0

    # k2: 用k1计算中点导数
    v_k2 = lin_v0 + k1_v * dt * 0.5
    p_k2 = p0 + k1_p * dt * 0.5
    f2 = compute_spring_force(p_k2, v_k2, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)
    k2_v = f2 / mass
    k2_p = v_k2

    # k3: 用k2计算中点导数
    v_k3 = lin_v0 + k2_v * dt * 0.5
    p_k3 = p0 + k2_p * dt * 0.5
    f3 = compute_spring_force(p_k3, v_k3, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)
    k3_v = f3 / mass
    k3_p = v_k3

    # k4: 用k3计算终点导数
    v_k4 = lin_v0 + k3_v * dt
    p_k4 = p0 + k3_p * dt
    f4 = compute_spring_force(p_k4, v_k4, p_anchor, gravity, mass, spring_stiffness, spring_rest_length, spring_damping)
    k4_v = f4 / mass
    k4_p = v_k4

    # 加权平均
    v_new = lin_v0 + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * dt / 6.0
    p_new = p0 + (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) * dt / 6.0

    # 更新输出状态
    body_q_out[i] = wp.transform(p=p_new, q=x0.q)
    body_qd_out[i] = wp.spatial_vector(v_new[0], v_new[1], v_new[2], v0[3], v0[4], v0[5])


class SolverExercise1ODESpring(SolverBase):
    """
    作业1:弹簧常微分方程求解器.

    实现多种时间积分方案:
    - 0: Analytic (解析解)
    - 1: Explicit Euler (显式欧拉)
    - 2: Semi-Implicit Euler (半隐式欧拉)
    - 3: Mid-Point (中点法)
    - 4: RK4 (四阶龙格-库塔)
    """

    def __init__(self, model: Model):
        super().__init__(model)

        self.method = 0  # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

        self.gravity = -9.81
        self.spring_rest_length = 5.0
        self.spring_stiffness = 5.0
        self.spring_damping = 0.1

        # 解析解需要存储初始位姿和速度
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

        # Mass of the body / 物体质量
        self.mass = 1.0

    def reset(self):
        """Reset solver state / 重置求解器状态."""
        self.time = 0.0
        self.initial_poses = wp.clone(self.model.body_q)
        self.initial_velocities = wp.clone(self.model.body_qd)

    def analytic(self, state_in: State, state_out: State, dt: float):
        """Analytic solution / 解析解."""
        self.time += dt
        wp.launch(
            analytic_spring_kernel,
            dim=self.model.body_count,
            inputs=[
                state_out.body_q,
                state_out.body_qd,
                self.initial_poses,
                self.initial_velocities,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.mass,
                self.time,
            ],
        )

    def explicit_euler(self, state_in: State, state_out: State, dt: float):
        """Explicit Euler method / 显式欧拉法."""
        wp.launch(
            explicit_euler_spring_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.mass,
                dt,
            ],
        )

    def semi_implicit_euler(self, state_in: State, state_out: State, dt: float):
        """Semi-Implicit Euler method / 半隐式欧拉法."""
        wp.launch(
            semi_implicit_euler_spring_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.mass,
                dt,
            ],
        )

    def mid_point(self, state_in: State, state_out: State, dt: float):
        """Mid-point method / 中点法."""
        wp.launch(
            mid_point_spring_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.mass,
                dt,
            ],
        )

    def rk4(self, state_in: State, state_out: State, dt: float):
        """RK4 method / 四阶龙格-库塔法."""
        wp.launch(
            rk4_spring_kernel,
            dim=self.model.body_count,
            inputs=[
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                self.gravity * wp.vec3(0.0, 0.0, 1.0),
                self.spring_rest_length,
                self.spring_stiffness,
                self.spring_damping,
                self.mass,
                dt,
            ],
        )

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
