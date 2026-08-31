# Reference projects

This project evaluates external work with pinned source and license records.
A reference may influence interfaces, tests, or design decisions; copied or
modified code is integrated only when its license and attribution are retained.

## BahiaRT MuJoCo base

The Python client started from the BahiaRT RCSSServerMJ base.  Its strongest
parts are the length-prefixed TCP client, perception parser, T1 motor mapping,
keyframe runner, and the original learned walking policy.  The current project
has since added the 55 m x 36 m seven-player field, role behaviours, set plays,
and a first kick motion.

Reference value: **high for protocol and basic motion bootstrapping, low for a
complete match strategy**.  The base has no core regression suite and leaves
several state-estimation and action-lifecycle contracts implicit.

## ApolloCodebase

ApolloCodebase was imported from a fresh online clone at revision
`71018c968969d6e55130b0e1987cd5b4f5c3b4df` and is now the authoritative C++
runtime under `runtime/apollo/`. It is licensed under GPL-3.0-or-later; the
combined repository carries that license and preserves upstream provenance.
Apollo3D's 2026 world title is corroborated by the [HKUST(GZ) Humanoid Computing Lab](https://hclab-gz.github.io/)
and [Offenburg University's competition report](https://www.hs-offenburg.de/en/hochschule/news/article/deutsche-teams-praegen-den-robocup-2026).

Its highest-value reference points are:

- a canonical team frame with our goal on negative x;
- typed high-level commands between decision and motion layers;
- a behaviour-tree decision pipeline and explicit role allocation;
- ball filtering, perception freshness, and teammate communication;
- obstacle-aware walk planning and legal set-play target generation;
- isolated ONNX walk/get-up runners and deployable runtime assets.

The upstream release did **not** expose a standalone kick command or kick
policy asset. My3D therefore ports its validated kick gates and lifecycle into
Apollo's typed command and motion layers while reusing Apollo's walk network.

Reference value: **very high for architecture, world modelling, navigation,
team coordination, and get-up recovery; medium for general low-level motion;
low for a directly reusable kick implementation**.

The earlier Python adapter remains useful as cross-implementation evidence,
but competition execution now uses Apollo's native 75-to-23 get-up runner.
Deployment archives include the license, notices, and corresponding modified
source. See `apollo-integration.md` for the exact provenance and reuse boundary.

## OtherTasks/cs61a

`OtherTasks/cs61a` is unrelated coursework and bundled third-party educational
code.  It has no reference value for RCSSServerMJ motion or match behaviour and
must be excluded from team packaging, linting, and test discovery.

## Local RoboCup2D team archives

The WSL reference directory `/home/win98/my_projects/rbc/teams` contains
Cyrus2DBase, two HELIOS base variants, TheMY, and CppDNN. Their main value is
not executable reuse in a 3D client but mature tactical decomposition:
cooperative action types, direct/leading/through candidate generation,
receiver-opponent arrival races, field evaluation, receive intentions, and
compact pass communication.

Reference value: **very high for planner concepts and tactical test cases;
low for direct runtime or physics reuse**. RoboCup2D dash/turn/kick cycles,
point-player kinematics, stamina, offside assumptions, and ball decay cannot
be transferred into the MuJoCo humanoid runtime. They must be replaced with
measured Apollo contact trajectories, rotation/locomotion reach time, fall
risk, observation freshness, and 3D rule geometry.

The local copies are extracted archives without Git metadata. Root license
files and individual file headers are also mixed across MIT, LGPL, and GPL.
The first strategy migration consequently uses a clean independent C++17
implementation and records the exact inspected-file hashes and license
boundary in `strategy-migration-implementation.md`.
