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

Run all twelve reference scenes and write complete bundles beneath the ignored
`evaluation/results/canonical/` directory:

```bash
uv run -m evaluation.run_canonical
```

Freeze deterministic schema-v3 baseline evidence (scene, metrics,
diagnostics, trajectory hash, device, Warp version, seed, and wall time):

```bash
uv run -m evaluation.freeze_baseline evaluation/baselines/schema_v3
```

Compare frozen pre-refactor and migrated post-refactor evidence using the
explicit penetration, cloth-quality, mass, and divergence tolerances:

```bash
uv run -m evaluation.compare_baselines \
  evaluation/baselines/schema_v2 \
  evaluation/baselines/schema_v3_migrated_v2 \
  --output evaluation/baselines/comparison.json
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

## Run ten black-box agent experiments

The ten natural-language tasks in `evaluation/agent_prompts.py` describe the
observable outcomes of the canonical scenes without exposing their Scene IR,
numeric acceptance assertions, or reference artifacts. The runner starts a
fresh `codex exec --ephemeral` process for every task and gives it independent
Codex configuration, scene storage, workspace, and output directories.

First prepare all inputs without invoking Codex:

```bash
uv run -m evaluation.run_agent_experiments \
  --dry-run
```

Run all ten experiments serially:

```bash
export WARP_CACHE_ROOT=/tmp/newton-warp-cache-$$
uv run -m evaluation.run_agent_experiments
```

When `--run-dir` is omitted, results are written beneath
`evaluation/results/agent-workflow/<timestamp>/`. Use `--run-dir` only when a
different explicit destination is needed.

Each experiment has a default timeout of 20 minutes. On timeout, the runner
terminates the complete Codex/MCP process group, records the case as failed,
and continues with the next experiment. Override the limit in seconds with
`--timeout`.

By default the runner copies only `~/.codex/auth.json` into each temporary
Codex home. It does not copy user configuration, sessions, or memories. To use
`OPENAI_API_KEY` instead, pass `--no-auth-copy`. Run one task with `--case`:

```bash
uv run -m evaluation.run_agent_experiments \
  --case 02_domino_wave \
  --run-dir /tmp/ca-agent-domino \
  --no-auth-copy
```

Each case retains its exact prompt, Codex JSONL event log, final message,
agent-created scene store, and output bundle. `summary.json` reports the Codex
exit code, elapsed time, and presence of every required artifact. This is a
black-box generation test: canonical scenes remain evaluator-only and are not
copied or mounted into the agent workspace.
