import numpy as np
import warp as wp

import newton
import newton.examples

class Exercise1ODESpring:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.viewer._paused = True
        self.args = args

        builder = newton.ModelBuilder()

        # Create a single body with a spherical shape
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -5.0), q=wp.quat_identity()),
            mass=1.0,
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=body, radius=1.0)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverExercise1ODESpring(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_state = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.initial_state.assign(self.state_0)

        self.viewer.set_model(self.model)

        self.method = 0 # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

    def gui(self, ui):
        _changed, self.method = ui.combo("Method", self.method, ["Analytic", "Explicit Euler", "Semi-Implicit Euler", "Mid-Point", "RK4"])
        if ui.button("Reset"):
            self.reset()

    def reset(self):
        self.viewer._paused = True
        self.sim_time = 0.0
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.solver.reset()

    def step(self):
        self.solver.method = self.method
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Exercise1ODESpring(viewer, args)
    newton.examples.run(example, args)
