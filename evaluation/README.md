# Prompt-to-video evaluation

This directory is the evaluation boundary. It contains black-box task prompts,
benchmark launchers, and generated results. It is not included in the Newton
wheel and must not be mounted in an agent's working directory during an
experiment.

The agent-facing surface consists only of:

- `ca_framework/` for the generic Scene IR and runtime;
- `mcp_server/` through its registered MCP command, not its source tree;
- `knowledge/skills/` for generic authoring and orchestration guidance;
- one prompt supplied by the external benchmark harness.

The evaluator owns `evaluation/`, starts a fresh agent conversation for every
prompt, assigns an independent scene workspace and result directory, and scores
the artifacts only after the agent exits. Never copy prior results or this
directory into an agent workspace.

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

## Run the 12-case black-box baselines

The private natural-language tasks in `evaluation/agent_prompts.py` describe
observable outcomes without exposing evaluator IR, numeric acceptance
assertions, or reference artifacts. The runner starts a
fresh `codex exec --ephemeral` process for every task and gives it independent
Codex configuration, scene storage, workspace, and output directories.

First prepare all inputs without invoking Codex:

```bash
uv run -m evaluation.run_agent_experiments \
  --dry-run
```

Run all cases for one workflow serially:

```bash
export WARP_CACHE_ROOT=/tmp/newton-warp-cache-$$
uv run -m evaluation.run_agent_experiments
```

Run the matched direct-Newton and MCP baselines serially:

```bash
export WARP_CACHE_ROOT=/tmp/newton-warp-cache-$$
uv run -m evaluation.run_agent_matrix --run-dir evaluation/results/final-matrix
```

## Run a targeted optimizer experiment

Optimization is deliberately separate from scene authoring. For a selected MCP
result, first ask an agent to inspect its baseline video and author one task
objective. Then run the existing Random, TPE, and CMA-ES suite on that fixed
scene. This is a targeted experiment; it does not rerun the 12-case agent
benchmark.

```bash
uv run -m evaluation.run_agent_objectives \
  --mcp-run-dir evaluation/results/final-matrix/mcp \
  --case <case-name> \
  --output-dir /tmp/ca-objectives

export WARP_CACHE_ROOT=/tmp/newton-warp-cache-$$
uv run -m evaluation.run_mcp_optimizer_suite \
  --mcp-run-dir evaluation/results/final-matrix/mcp \
  --objective-dir /tmp/ca-objectives \
  --case <case-name> \
  --output-dir /tmp/ca-optimizer-results
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
`OPENAI_API_KEY` instead, pass `--no-auth-copy`. To run one selected task,
pass its evaluator case name to `--case`:

```bash
uv run -m evaluation.run_agent_experiments \
  --case <case-name> \
  --run-dir /tmp/ca-agent-run \
  --no-auth-copy
```

Each case retains its exact prompt, Codex JSONL event log, final message,
agent-created scene store where applicable, and output bundle. `summary.json`
reports the selected workflow/optimizer, Codex exit code, elapsed time, and
presence of every required artifact. The direct baseline receives only a copied
Newton module; it has neither MCP tools nor `ca_framework` source.
