# Contributing

Development is performed in WSL2 Ubuntu 22.04.  Keep the competition runtime
and reinforcement-learning environment separate: `my3d-team` is Python 3.13;
`my3d-rl` is Python 3.12 with pinned GPU packages.

## Runtime changes

Before opening a pull request:

```bash
conda activate my3d-team
black --check mujococodebase run_player.py scripts tests training
flake8 mujococodebase run_player.py scripts tests training
python -m compileall -q mujococodebase run_player.py scripts
bash -n scripts/*.sh start7v7.sh
pytest -q
```

Changes to movement, motor output, perception, or match state must also pass:

```bash
export MY3D_PYTHON="$CONDA_PREFIX/bin/python"
export RCSSSERVERMJ_BIN="$HOME/.local/bin/rcssservermj"
scripts/run_acceptance_match.sh 600
```

Do not weaken the finite motor checks, role ownership rules, or deterministic
shutdown assertions to make an acceptance run pass.

## Training changes

Use `training/README.md` and keep generated checkpoints, videos, logs, and
experiment caches outside Git.  A policy cannot enter the runtime without the
three-seed held-out evaluation, ONNX parity result, and RCSSServerMJ acceptance
evidence specified in `docs/rl-training-plan.md`.

## Pull requests

- Keep commits focused and explain observable behavior changes.
- Add regression tests for defects and state-machine transitions.
- Record external source revisions and licences before reusing assets.
- Never commit credentials, local environment files, generated models, or
  match logs.
- Update the runbook and validation record when operational behavior changes.
