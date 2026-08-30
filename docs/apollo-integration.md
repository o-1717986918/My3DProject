# Apollo get-up integration

## Scope

`external/ApolloCodebase` is pinned at commit
`71018c968969d6e55130b0e1987cd5b4f5c3b4df`. The Python adapter in
`mujococodebase/skills/external/apollo_get_up.py` loads only Apollo's
`assets/networks/getup/policy.onnx`; it does not launch the C++ team or replace
this project's decision controller, walking policy, or kick implementation.

The adapter matches the published C++ policy contract:

- input: 75 floats containing body gyro, projected gravity, 23 joint position
  errors, 23 joint velocities, and the previous 23-value action;
- output: 23 clipped relative joint actions;
- target: current joint position plus `0.6 * action`;
- lifecycle: reset action history on entry or after six seconds, and finish only
  after an upright, low-angular-speed pose remains stable for 0.35 seconds.

The decision layer does not overwrite head joints while `GetUp` owns the motor
packet. A 0.6-second zero-velocity walk hold smooths the handoff back to normal
play. If model loading or inference fails, the adapter disables itself for that
process and initializes the built-in keyframe recovery immediately.

## Runtime selection

Initialize the pinned model and use the default learned backend:

```bash
git submodule update --init --recursive
conda activate my3d-team
scripts/run_recovery_scenario.sh front
```

Override the model location:

```bash
export MY3D_APOLLO_GETUP_MODEL=/absolute/path/to/policy.onnx
```

Force the independent fallback:

```bash
export MY3D_GETUP_BACKEND=keyframe
```

Run all deterministic orientations:

```bash
for pose in front back left right; do
    scripts/run_recovery_scenario.sh "$pose" 600
done
```

## Licence boundary

ApolloCodebase and its policy assets declare GPL-3.0-or-later. The main project
currently has no repository-wide licence declaration. The default PyInstaller
script therefore does not copy Apollo's model into its binary/archive.

Local source-tree execution can load the model from the separately checked-out
submodule. Before distributing a package that contains both this client and the
Apollo asset, the project owners must choose and document a GPL-compatible
licensing/source-delivery approach, include Apollo's licence and notices, and
satisfy the corresponding-source obligations. This is a release decision, not
a runtime defect, and it must not be bypassed by silently embedding the model.
