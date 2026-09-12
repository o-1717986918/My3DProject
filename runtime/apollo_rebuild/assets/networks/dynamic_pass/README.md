# Experimental dynamic-pass selector

This directory contains an opt-in selector for one narrow action condition:
straight `2.0 m` pass, requested ball speed `1.43 m/s`, desired arrival speed
`0.8 m/s`. It is enabled by default under continued server evaluation and can
be disabled cleanly for A/B matches; this is not evidence of general passing,
dribbling, or shooting capability.

- Model: `selector.onnx`
- SHA-256: `ab026000494baea0b00fecabe84300966e5788869f490d9a0ca4c58e73aa7a3c`
- Input/output: `[1,98] -> [1,10]`
- Frozen release: probability at least `0.96` for two consecutive frames
- Prototype rollout IDs, in output order: `302, 117, 4, 84, 99, 107, 43, 79, 65, 17`
- Training manifest: `/home/win98/rl_runs/kick-switch-selector-all-fallback90-s10806/selector.json`
- Source git revision recorded by that manifest: `13e1d75d18303aac8fed253ec1abbd2449ce8937`

The C++ executor uses the accepted fourteen-parameter rows from the source
prototype manifests and reproduces the training composition: decode the smooth
trajectory, apply the per-joint kick scale, then overlay it on the current
Apollo Walk target. The previous procedural runner did not have this contract.

Independent exact-CPU reports are stored under
`/home/win98/rl_runs/kick-switch-selector-independent-s10917` and
`/home/win98/rl_runs/kick-switch-selector-independent-s10923`. Two real
RCSSServerMJ fixed-near-ball comparisons triggered the action without a fall,
but its 1.26--1.39 m lateral miss did not establish a usable straight pass.
The selector is a default-on, explicitly reversible evaluation capability, not
a promoted general ball-action model. See
`docs/baseline-first-discrete-capability-roadmap.md` for current server results.
