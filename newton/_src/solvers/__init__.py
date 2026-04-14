# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags
from .implicit_mpm import SolverImplicitMPM
from .mujoco import SolverMuJoCo
from .semi_implicit import SolverSemiImplicit
from .solver import SolverBase
from .style3d.solver_style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD
from .ca_exercises import SolverExercise1ODECannonBall
from .ca_exercises import SolverExercise1ODESpring
from .ca_exercises import SolverExercise2ConstraintWire
from .ca_exercises import SolverExercise2ConstraintPendulum
from .ca_exercises import SolverExercise3RigidBody

__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
    "SolverExercise1ODECannonBall",
    "SolverExercise1ODESpring",
    "SolverExercise2ConstraintWire",
    "SolverExercise2ConstraintPendulum",
    "SolverExercise3RigidBody",
]
