# Apollo motion-policy migration contract

## Current decision

Keep Apollo's shipped walk and get-up policies as the competition baseline,
reuse the validated My3D kick controller immediately, and retrain future
running policies against an Apollo-native deployment contract. Do not insert
the current v4 ONNX file into the walk asset path.

## Why the current v4 model is not a drop-in replacement

| Boundary | Apollo walk baseline | My3D v4 candidate |
|---|---:|---:|
| Actor input | `[1, 78]` | `[1, 80]` |
| Actor output | `[1, 23]` | `[1, 23]` |
| Decoder | self-contained relative action, scale 0.25 | residual over phase-indexed reference, scale 0.15 |
| External reference | none | required 34x23 NPZ |
| Distribution | source asset | reference is local-only/restricted |
| Release state | validated baseline | training-only/rejected for running |

Matching output width does not imply compatible semantics. Loading v4 through
`WalkRunner` would fail the input contract; bypassing that check would produce
incorrect targets because the reference decoder and phase state are missing.

## Assets already reused

1. Apollo's online C++ behavior tree and policy runners are now the main stack.
2. Apollo's two learned policy assets are retained byte-for-byte.
3. My3D's stable kick engagement window, duration, cooldown, and recovery hold
   were ported into the C++ decision/motion layers.
4. The Python training environments, contracts, evaluation tools, reference
   importers, and research locks remain available for the next training round.

## Efficient next training target

Train a self-contained `apollo_run_v1` policy with an explicitly versioned
contract:

- input `[1, 78]` using Apollo's exact `WalkRunner::build_observation` order;
- output `[1, 23]` using Apollo's exact joint order and relative-action decoder;
- control at 50 Hz with the existing velocity command and action history;
- curriculum from stand/walk tracking to faster forward motion, turning,
  stopping, pushes, ball approach, and multi-agent perturbations;
- teacher/reference motion may guide training, but no restricted reference may
  be required at inference time;
- export a single ONNX actor plus manifest/hash and compare C++/Python outputs
  on a fixed observation corpus.

This route keeps the mature runtime and removes a second C++ decoder, reducing
both implementation time and deployment risk. If an 80-value residual policy
is still scientifically valuable, integrate it later as a separately named
runner with its own reference format and fallback; never overload `walk.onnx`.

## Promotion gates

1. **Contract:** dimensions, ordering, normalization, decoder, and hashes pass.
2. **Parity:** ONNX Runtime C++ and training evaluator agree within tolerance.
3. **Dynamics:** forward/turn/stop/push suites beat the baseline without a
   higher fall rate across at least three seeds.
4. **Server:** single-player RCSS tests show valid bounded motors and recovery.
5. **Match:** two independent strict 7v7 runs pass; then a visual run is
   reviewed for collisions, oscillation, ball ownership, and recovery.
6. **Fallback:** any model/shape/non-finite/posture failure returns to Apollo's
   baseline in the same cycle.
