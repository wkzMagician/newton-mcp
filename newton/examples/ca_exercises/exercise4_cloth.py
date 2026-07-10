import time

import warp as wp

import newton
import newton.examples


class Exercise4Cloth:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Cloth grid parameters
        self.grid_n = 16
        self.grid_m = 16
        self.cell_size = 0.15
        self.particle_mass = 1.0

        # Spring parameters
        self.k_struct = 500.0
        self.k_bend = 60.0
        self.k_shear = 50.0
        self.damping = 10.0

        # Integration method (0=explicit, 1=implicit, 2=semi-implicit)
        self.method = 0

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args

        builder = newton.ModelBuilder()

        # Build cloth grid with particles and springs
        self._build_cloth_grid(builder)

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverExercise4Cloth(
            self.model,
            method=self.method,
            params={
                "grid_n": self.grid_n,
                "grid_m": self.grid_m,
                "cell_size": self.cell_size,
                "particle_mass": self.particle_mass,
                "k_struct": self.k_struct,
                "k_bend": self.k_bend,
                "k_shear": self.k_shear,
                "damping": self.damping,
            },
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_state = self.model.state()
        self.initial_state.assign(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = True

    def _build_cloth_grid(self, builder):
        # Add particles
        for i in range(self.grid_n):
            for j in range(self.grid_m):
                pos = wp.vec3(i * self.cell_size, j * self.cell_size, 0.0)
                vel = wp.vec3(0.0, 0.0, 0.0)
                builder.add_particle(pos, vel, self.particle_mass)

        # Add triangles for rendering
        for i in range(self.grid_n - 1):
            for j in range(self.grid_m - 1):
                p00 = i * self.grid_m + j
                p10 = (i + 1) * self.grid_m + j
                p01 = i * self.grid_m + (j + 1)
                p11 = (i + 1) * self.grid_m + (j + 1)
                builder.add_triangle(p00, p10, p01)
                builder.add_triangle(p10, p11, p01)

    def gui(self, ui):
        if ui.button("Reset"):
            self.reset()

        # Method selector
        _changed, self.method = ui.combo(
            "Method",
            self.method,
            ["Explicit Euler", "Implicit Euler", "Semi-Implicit Euler"],
        )
        self.solver.method = self.method

    def reset(self):
        self.sim_time = 0.0
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.solver.reset()
        self.viewer._paused = True

    def step(self):
        start_time = time.time()
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        elapsed = time.time() - start_time
        if elapsed < self.frame_dt:
            time.sleep(self.frame_dt - elapsed)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Exercise4Cloth(viewer, args)
    newton.examples.run(example, args)
