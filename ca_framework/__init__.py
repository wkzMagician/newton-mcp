# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Agent-facing scene authoring framework built on Newton."""

from .canonical import canonical_scenes
from .scene import Scene, SceneExecutorLocal, SceneStore

__all__ = ["Scene", "SceneExecutorLocal", "SceneStore", "canonical_scenes"]
