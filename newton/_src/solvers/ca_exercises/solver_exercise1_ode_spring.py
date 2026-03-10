import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase

# TODO: Implement the kernels here
# You can refer to the CannonBall example for some guidance


class SolverExercise1ODESpring(SolverBase):
    def __init__(self, model: Model):
        super().__init__(model)

        self.method = 0 # 0: Analytic, 1: Explicit Euler, 2: Semi-Implicit Euler, 3: Mid-Point, 4: RK4

        self.gravity = -9.81
        self.spring_rest_length = 5.0
        self.spring_stiffness = 5.0
        self.spring_damping = 0.1

        # TODO: You can add additional attributes here if needed
        # Notes: the ball is initially at (0, 0, -spring_rest_length) with zero velocity, 
        # and the spring is attached to the origin (0, 0, 0). You can change the parameters
        # above to attain better results

    def reset(self):
        # TODO: reset any additional attributes you added in the constructor here
        
        pass

    def analytic(self, state_in: State, state_out: State, dt: float):
        # TODO: launch the analytic kernel here

        pass

    def explicit_euler(self, state_in: State, state_out: State, dt: float):
        # TODO: launch the explicit Euler kernel here

        pass

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
