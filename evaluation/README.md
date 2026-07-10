# Prompt-to-video evaluation

This directory is the evaluation boundary. It contains reference scenes,
acceptance assertions, benchmark launchers, and generated results. It is not
included in the Newton wheel and must not be mounted in an agent's working
directory during an experiment.

The agent-facing surface consists only of:

- `ca_framework/` for the generic Scene IR and runtime;
- `mcp_server/` through its registered MCP command, not its source tree;
- `knowledge/skills/` for generic authoring and orchestration guidance;
- one prompt supplied by the external benchmark harness.

The evaluator owns `evaluation/`, starts a fresh agent conversation for every
prompt, assigns an independent scene workspace and result directory, and scores
the artifacts only after the agent exits. Never copy reference scenes, expected
metrics, prior results, or this directory into an agent workspace.

## Run the hidden canonical acceptance suite

From the repository root:

```bash
export WARP_CACHE_ROOT=/tmp/newton-warp-cache-$$
uv run --extra dev -m unittest evaluation.tests.test_canonical_scenes
```

Run all ten reference scenes and write complete bundles beneath the ignored
`evaluation/results/canonical/` directory:

```bash
uv run -m evaluation.run_canonical
```

Run one scene:

```bash
uv run -m evaluation.run_canonical --scene 08_liquid_pour
```

## Start an uncontaminated agent

Create an empty directory outside the repository and start the agent there. The
MCP server may execute code from this checkout, but its scene store and outputs
must point to directories owned by that individual run.

```bash
run_id=run-001
repo=/home/wukunzhen/Codes/ca-framework-2026
mkdir -p "$repo/evaluation/workspaces/$run_id" "$repo/evaluation/results/$run_id"
cd "$repo/evaluation/workspaces/$run_id"
codex
```

Register the MCP using an absolute project path. Do this once outside the
measured agent conversation:

```bash
repo=/home/wukunzhen/Codes/ca-framework-2026
codex mcp add ca-scene -- \
  env CA_SCENE_WORKSPACE="$repo/evaluation/workspaces/current/scenes" \
  uv run --project "$repo/mcp_server" ca-scene-mcp
```

For strict multi-run evaluation, generate a separate Codex home/MCP
configuration per run so `CA_SCENE_WORKSPACE` cannot leak scenes between runs.
Only install the generic skills from `knowledge/skills`; do not create a skill
that imports or describes anything in this directory.
