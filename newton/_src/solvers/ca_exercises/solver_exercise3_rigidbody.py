# pyright: reportInvalidTypeForm=false


import numpy as np
import warp as wp

from ...core.types import override
from ...geometry.broad_phase_common import check_aabb_overlap
from ...geometry.narrow_phase import NarrowPhase
from ...geometry.types import GeoType
from ...sim import Contacts, Control, Model, State
from ...sim.collide import (
    ContactWriterData,
    compute_shape_aabbs,
    prepare_geom_data_kernel,
    write_contact,
)
from ..solver import SolverBase


@wp.kernel
def broadphase_dynamic_aabb_tree_kernel(
    bvh_id: wp.uint64,
    shape_aabb_lower: wp.array(dtype=wp.vec3),
    shape_aabb_upper: wp.array(dtype=wp.vec3),
    candidate_pair: wp.array(dtype=wp.vec2i),
    candidate_pair_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()

    lower_i = shape_aabb_lower[i]
    upper_i = shape_aabb_upper[i]

    query = wp.bvh_query_aabb(bvh_id, lower_i, upper_i)
    j = int(0)

    while wp.bvh_query_next(query, j):
        if j <= i:
            continue

        lower_j = shape_aabb_lower[j]
        upper_j = shape_aabb_upper[j]

        if not check_aabb_overlap(lower_i, upper_i, 0.0, lower_j, upper_j, 0.0):
            continue

        pair = wp.vec2i(i, j)
        pair_id = wp.atomic_add(candidate_pair_count, 0, 1)
        candidate_pair[pair_id] = pair


@wp.kernel
def integrate_bodies_kernel(
    body_count: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_q_out: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    gravity: wp.vec3,
    dt: float,
    angular_damping: float,
):
    i = wp.tid()
    if i >= body_count:
        return

    q_new = body_q[i]
    qd_new = body_qd[i]

    inv_mass = body_inv_mass[i]
    x = wp.transform_get_translation(q_new)
    r = wp.transform_get_rotation(q_new)
    w = wp.spatial_top(qd_new)
    v = wp.spatial_bottom(qd_new)

    if inv_mass > 0.0:
        v = v + gravity * dt
        x = x + v * dt

        w = w - w * angular_damping * dt
        r = wp.normalize(r + wp.quat(w, 0.0) * r * 0.5 * dt)

        q_new = wp.transform(x, r)
        qd_new = wp.spatial_vector(w, v)

    body_q_out[i] = q_new
    body_qd_out[i] = qd_new


@wp.func
def world_inv_inertia(q: wp.transform, inv_inertia_body: wp.mat33) -> wp.mat33:
    r = wp.quat_to_matrix(wp.transform_get_rotation(q))
    return r * inv_inertia_body * wp.transpose(r)


@wp.kernel
def sequential_impulse_contacts_kernel(
    dt: float,
    baumgarte: float,
    shape_body: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    shape_material_mu: wp.array(dtype=float),
    shape_material_restitution: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),  # Current body transforms
    body_qd: wp.array(dtype=wp.spatial_vector),  # Current body velocities (linear, angular)
    body_qd_old: wp.array(dtype=wp.spatial_vector),  # Old body velocities
    contact_count: wp.array(dtype=wp.int32),
    contact_max: int,
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_lambda_n: wp.array(dtype=float),
    contact_lambda_t: wp.array(dtype=float),
):
    # Single-thread loop for true sequential (Gauss-Seidel) updates.
    if wp.tid() != 0:
        return

    count = wp.min(contact_count[0], contact_max)
    for c in range(count):
        shape_a = contact_shape0[c]
        shape_b = contact_shape1[c]

        body_a = shape_body[shape_a]
        body_b = shape_body[shape_b]

        if body_a < 0 and body_b < 0:
            continue

        xform_a = wp.transform_identity()
        xform_b = wp.transform_identity()
        if body_a >= 0:
            xform_a = body_q[body_a]
        if body_b >= 0:
            xform_b = body_q[body_b]

        if body_a == body_b:
            continue

        # Contact point in world space
        p_a = wp.transform_point(xform_a, contact_point0[c] + contact_offset0[c])
        p_b = wp.transform_point(xform_b, contact_point1[c] + contact_offset1[c])

        n = contact_normal[c]  # ? 符号

        com_a = wp.vec3(0.0)
        com_b = wp.vec3(0.0)
        if body_a >= 0:
            com_a = wp.transform_point(xform_a, body_com[body_a])
        if body_b >= 0:
            com_b = wp.transform_point(xform_b, body_com[body_b])

        r_a = p_a - com_a
        r_b = p_b - com_b

        v_a_old = wp.vec3(0.0)
        w_a_old = wp.vec3(0.0)
        if body_a >= 0:
            qd_a_old = body_qd_old[body_a]
            w_a_old = wp.spatial_top(qd_a_old)
            v_a_old = wp.spatial_bottom(qd_a_old)
        v_a_cp_old = v_a_old + wp.cross(w_a_old, r_a)

        v_b_old = wp.vec3(0.0)
        w_b_old = wp.vec3(0.0)
        if body_b >= 0:
            qd_b_old = body_qd_old[body_b]
            w_b_old = wp.spatial_top(qd_b_old)
            v_b_old = wp.spatial_bottom(qd_b_old)
        v_b_cp_old = v_b_old + wp.cross(w_b_old, r_b)

        rel_v_old = v_a_cp_old - v_b_cp_old
        rel_n_old = wp.dot(rel_v_old, n)

        separation = wp.dot(n, p_a - p_b)
        penetration = wp.max(0.0, -separation)

        # 两种材料,取开方
        restitution = wp.sqrt(shape_material_restitution[shape_a] * shape_material_restitution[shape_b])

        target_vn = baumgarte * penetration / dt - restitution * rel_n_old

        denom_n = 0.0

        zero_inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        inv_inertia_a = zero_inertia
        inv_inertia_b = zero_inertia

        if body_a >= 0:
            rxn_a = wp.cross(r_a, n)
            inv_inertia_a = world_inv_inertia(body_q[body_a], body_inv_inertia[body_a])
            denom_n += body_inv_mass[body_a] + wp.dot(n, wp.cross(inv_inertia_a * rxn_a, r_a))

        if body_b >= 0:
            rxn_b = wp.cross(r_b, n)
            inv_inertia_b = world_inv_inertia(body_q[body_b], body_inv_inertia[body_b])
            denom_n += body_inv_mass[body_b] + wp.dot(n, wp.cross(inv_inertia_b * rxn_b, r_b))

        v_a = wp.vec3(0.0)
        w_a = wp.vec3(0.0)
        if body_a >= 0:
            qd_a = body_qd[body_a]
            w_a = wp.spatial_top(qd_a)
            v_a = wp.spatial_bottom(qd_a)

        v_b = wp.vec3(0.0)
        w_b = wp.vec3(0.0)
        if body_b >= 0:
            qd_b = body_qd[body_b]
            w_b = wp.spatial_top(qd_b)
            v_b = wp.spatial_bottom(qd_b)

        rel_v = (v_a + wp.cross(w_a, r_a)) - (v_b + wp.cross(w_b, r_b))
        vn = wp.dot(rel_v, n)

        lambda_n_old = contact_lambda_n[c]
        delta_lambda_n = (target_vn - vn) / denom_n
        lambda_n_new = wp.max(lambda_n_old + delta_lambda_n, 0.0)
        delta_lambda_n = lambda_n_new - lambda_n_old
        contact_lambda_n[c] = lambda_n_new

        impulse_n = n * delta_lambda_n

        if body_a >= 0:
            qd_a = body_qd[body_a]
            w_a_old_cur = wp.spatial_top(qd_a)
            v_a_old_cur = wp.spatial_bottom(qd_a)
            v_a_new = v_a_old_cur + impulse_n * body_inv_mass[body_a]
            w_a_new = w_a_old_cur + inv_inertia_a * wp.cross(r_a, impulse_n)
            body_qd[body_a] = wp.spatial_vector(w_a_new, v_a_new)

        if body_b >= 0:
            qd_b = body_qd[body_b]
            w_b_old_cur = wp.spatial_top(qd_b)
            v_b_old_cur = wp.spatial_bottom(qd_b)
            v_b_new = v_b_old_cur - impulse_n * body_inv_mass[body_b]
            w_b_new = w_b_old_cur - inv_inertia_b * wp.cross(r_b, impulse_n)
            body_qd[body_b] = wp.spatial_vector(w_b_new, v_b_new)

        v_a = wp.vec3(0.0)
        if body_a >= 0:
            qd_a = body_qd[body_a]
            v_a = wp.spatial_bottom(qd_a) + wp.cross(wp.spatial_top(qd_a), r_a)

        v_b = wp.vec3(0.0)
        if body_b >= 0:
            qd_b = body_qd[body_b]
            v_b = wp.spatial_bottom(qd_b) + wp.cross(wp.spatial_top(qd_b), r_b)

        rel_v = v_a - v_b
        vt = rel_v - n * wp.dot(rel_v, n)  # 切向相对速度
        vt_len = wp.length(vt)
        t = vt / (vt_len + 1e-6)  # 切向向量
        denom_t = 0.0

        if body_a >= 0:
            rxt_a = wp.cross(r_a, t)
            denom_t += body_inv_mass[body_a] + wp.dot(t, wp.cross(inv_inertia_a * rxt_a, r_a))

        if body_b >= 0:
            rxt_b = wp.cross(r_b, t)
            denom_t += body_inv_mass[body_b] + wp.dot(t, wp.cross(inv_inertia_b * rxt_b, r_b))

        mu = wp.sqrt(shape_material_mu[shape_a] * shape_material_mu[shape_b])  # 取average?
        # max_friction = mu * lambda_n_new # ? 单次冲量还是总冲量?
        max_friction = mu * contact_lambda_n[c]
        lambda_t_old = contact_lambda_t[c]
        delta_lambda_t = -vt_len / denom_t
        lambda_t_new = wp.clamp(lambda_t_old + delta_lambda_t, -max_friction, max_friction)
        delta_lambda_t = lambda_t_new - lambda_t_old
        contact_lambda_t[c] = lambda_t_new

        impulse_t = t * delta_lambda_t

        if body_a >= 0:
            qd_a = body_qd[body_a]
            w_a_old_cur = wp.spatial_top(qd_a)
            v_a_old_cur = wp.spatial_bottom(qd_a)
            v_a_new = v_a_old_cur + impulse_t * body_inv_mass[body_a]
            w_a_new = w_a_old_cur + inv_inertia_a * wp.cross(r_a, impulse_t)
            body_qd[body_a] = wp.spatial_vector(w_a_new, v_a_new)

        if body_b >= 0:
            qd_b = body_qd[body_b]
            w_b_old_cur = wp.spatial_top(qd_b)
            v_b_old_cur = wp.spatial_bottom(qd_b)
            v_b_new = v_b_old_cur - impulse_t * body_inv_mass[body_b]
            w_b_new = w_b_old_cur - inv_inertia_b * wp.cross(r_b, impulse_t)
            body_qd[body_b] = wp.spatial_vector(w_b_new, v_b_new)


class SolverExercise3RigidBody(SolverBase):
    def __init__(
        self,
        model: Model,
        iterations: int = 16,
        baumgarte: float = 0.2,
        angular_damping: float = 0.6,
    ):
        super().__init__(model)

        self.gravity = wp.vec3(0.0, 0.0, -9.81)

        self.iterations = iterations
        self.angular_damping = angular_damping
        self.baumgarte = baumgarte

        self.shape_count = model.shape_count
        self.max_candidate_pairs = (self.shape_count * (self.shape_count - 1)) // 2

        self.shape_aabb_lower = wp.zeros(self.shape_count, dtype=wp.vec3, device=model.device)
        self.shape_aabb_upper = wp.zeros(self.shape_count, dtype=wp.vec3, device=model.device)
        self.candidate_pairs = wp.zeros(self.max_candidate_pairs, dtype=wp.vec2i, device=model.device)
        self.candidate_pair_count = wp.zeros(1, dtype=wp.int32, device=model.device)
        self.geom_data = wp.zeros(self.shape_count, dtype=wp.vec4, device=model.device)
        self.geom_transform = wp.zeros(self.shape_count, dtype=wp.transform, device=model.device)

        self._bvh = None

        shape_types = model.shape_type.numpy() if model.shape_type is not None else np.zeros((0,), dtype=np.int32)
        has_meshes = bool((shape_types == int(GeoType.MESH)).any())
        has_heightfields = bool((shape_types == int(GeoType.HFIELD)).any())

        self.narrow_phase = NarrowPhase(
            max_candidate_pairs=self.max_candidate_pairs,
            max_triangle_pairs=1_000_000,
            reduce_contacts=True,
            device=model.device,
            shape_aabb_lower=self.shape_aabb_lower,
            shape_aabb_upper=self.shape_aabb_upper,
            contact_writer_warp_func=write_contact,
            shape_voxel_resolution=model._shape_voxel_resolution,
            hydroelastic_sdf=None,
            has_meshes=has_meshes,
            has_heightfields=has_heightfields,
        )

        self.contact_lambda_n = wp.zeros(self.max_candidate_pairs, dtype=float, device=model.device)
        self.contact_lambda_t = wp.zeros(self.max_candidate_pairs, dtype=float, device=model.device)

    def _integrate_bodies(
        self, model: Model, state_in: State, state_out: State, dt: float, angular_damping: float
    ) -> None:
        wp.launch(
            kernel=integrate_bodies_kernel,
            dim=model.body_count,
            inputs=[
                model.body_count,
                state_in.body_q,
                state_in.body_qd,
                state_out.body_q,
                state_out.body_qd,
                model.body_inv_mass,
                self.gravity,
                dt,
                angular_damping,
            ],
            device=model.device,
        )

    def _detect_contacts(self, state: State, contacts: Contacts) -> None:
        contacts.clear()
        self.candidate_pair_count.zero_()

        wp.launch(
            kernel=compute_shape_aabbs,
            dim=self.shape_count,
            inputs=[
                state.body_q,
                self.model.shape_transform,
                self.model.shape_body,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_collision_radius,
                self.model.shape_source_ptr,
                self.model.shape_margin,
                self.model.shape_gap,
            ],
            outputs=[self.shape_aabb_lower, self.shape_aabb_upper],
            device=self.device,
        )

        if self._bvh is None:
            self._bvh = wp.Bvh(self.shape_aabb_lower, self.shape_aabb_upper, groups=self.model.shape_world)
        else:
            self._bvh.refit()

        wp.launch(
            kernel=broadphase_dynamic_aabb_tree_kernel,
            dim=self.shape_count,
            inputs=[
                self._bvh.id,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
                self.candidate_pairs,
                self.candidate_pair_count,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=prepare_geom_data_kernel,
            dim=self.shape_count,
            inputs=[
                self.model.shape_transform,
                self.model.shape_body,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_margin,
                state.body_q,
            ],
            outputs=[self.geom_data, self.geom_transform],
            device=self.device,
        )

        writer_data = ContactWriterData()
        writer_data.contact_max = contacts.rigid_contact_max
        writer_data.body_q = state.body_q
        writer_data.shape_body = self.model.shape_body
        writer_data.shape_gap = self.model.shape_gap
        writer_data.contact_count = contacts.rigid_contact_count
        writer_data.out_shape0 = contacts.rigid_contact_shape0
        writer_data.out_shape1 = contacts.rigid_contact_shape1
        writer_data.out_point0 = contacts.rigid_contact_point0
        writer_data.out_point1 = contacts.rigid_contact_point1
        writer_data.out_offset0 = contacts.rigid_contact_offset0
        writer_data.out_offset1 = contacts.rigid_contact_offset1
        writer_data.out_normal = contacts.rigid_contact_normal
        writer_data.out_margin0 = contacts.rigid_contact_margin0
        writer_data.out_margin1 = contacts.rigid_contact_margin1
        writer_data.out_tids = contacts.rigid_contact_tids
        writer_data.out_stiffness = contacts.rigid_contact_stiffness
        writer_data.out_damping = contacts.rigid_contact_damping
        writer_data.out_friction = contacts.rigid_contact_friction

        self.narrow_phase.launch_custom_write(
            candidate_pair=self.candidate_pairs,
            candidate_pair_count=self.candidate_pair_count,
            shape_types=self.model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=self.model.shape_source_ptr,
            sdf_data=self.model.sdf_data,
            shape_sdf_index=self.model.shape_sdf_index,
            shape_gap=self.model.shape_gap,
            shape_collision_radius=self.model.shape_collision_radius,
            shape_flags=self.model.shape_flags,
            shape_collision_aabb_lower=self.model.shape_collision_aabb_lower,
            shape_collision_aabb_upper=self.model.shape_collision_aabb_upper,
            shape_voxel_resolution=self.narrow_phase.shape_voxel_resolution,
            shape_heightfield_data=self.model.shape_heightfield_data,
            heightfield_elevation_data=self.model.heightfield_elevation_data,
            writer_data=writer_data,
            device=self.device,
        )

    def _solve_contacts(self, state: State, contacts: Contacts, dt: float) -> None:
        self.contact_lambda_n.zero_()
        self.contact_lambda_t.zero_()
        old_body_qd = wp.clone(state.body_qd)
        for _ in range(self.iterations):
            wp.launch(
                kernel=sequential_impulse_contacts_kernel,
                dim=1,
                inputs=[
                    dt,
                    self.baumgarte,
                    self.model.shape_body,
                    self.model.body_com,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                    self.model.shape_material_mu,
                    self.model.shape_material_restitution,
                    state.body_q,
                    state.body_qd,
                    old_body_qd,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_max,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_offset0,
                    contacts.rigid_contact_offset1,
                    contacts.rigid_contact_normal,
                    self.contact_lambda_n,
                    self.contact_lambda_t,
                ],
                device=self.device,
            )

    def reset(self):
        self.contact_lambda_n.zero_()
        self.contact_lambda_t.zero_()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        if self.model.body_count == 0:
            return None

        # 1) External forces + gravity integration.
        self._integrate_bodies(self.model, state_in, state_out, dt, angular_damping=self.angular_damping)

        my_contacts = self.model.contacts()

        # 2) Collision detection: dynamic AABB tree broad phase + Newton narrow phase.
        self._detect_contacts(state_out, my_contacts)

        # 3) Sequential impulse iterations on rigid contact constraints.
        self._solve_contacts(state_out, my_contacts, dt)

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        # Contacts are written directly in _detect_contacts().
        _ = contacts
