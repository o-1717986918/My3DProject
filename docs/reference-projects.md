# Reference projects

This project keeps external code separate from the Python team runtime.  A
reference may influence interfaces, tests, or design decisions, but code is not
copied unless its licence is compatible and attribution is preserved.

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

ApolloCodebase is included as the Git submodule `external/ApolloCodebase` at a
pinned revision (`71018c968969d6e55130b0e1987cd5b4f5c3b4df`). It is licensed
under GPL-3.0-or-later and remains a separate source tree. Apollo3D's 2026 world
title is corroborated by the [HKUST(GZ) Humanoid Computing Lab](https://hclab-gz.github.io/)
and [Offenburg University's competition report](https://www.hs-offenburg.de/en/hochschule/news/article/deutsche-teams-praegen-den-robocup-2026).

Its highest-value reference points are:

- a canonical team frame with our goal on negative x;
- typed high-level commands between decision and motion layers;
- a behaviour-tree decision pipeline and explicit role allocation;
- ball filtering, perception freshness, and teammate communication;
- obstacle-aware walk planning and legal set-play target generation;
- isolated ONNX walk/get-up runners and deployable runtime assets.

The public release does **not** expose a standalone kick command or kick policy
asset.  Its public high-level motion variants are beam, walk, get-up, and
neutral, so it cannot by itself provide this team's scoring action.

Reference value: **very high for architecture, world modelling, navigation,
team coordination, and get-up recovery; medium for general low-level motion;
low for a directly reusable kick implementation**.

The Python runtime now has a narrow adapter that can load Apollo's get-up ONNX
asset directly from the submodule. The 75-value observation and 23-value
relative joint-action contract were validated against Apollo's C++ runner and
then tested in the real simulator from four fall directions. No Apollo source
file or model is copied into the Python package or the default PyInstaller
archive. See `apollo-integration.md` for the exact runtime and licensing
boundary; distributing a combined package requires explicit GPL compliance.

## OtherTasks/cs61a

`OtherTasks/cs61a` is unrelated coursework and bundled third-party educational
code.  It has no reference value for RCSSServerMJ motion or match behaviour and
must be excluded from team packaging, linting, and test discovery.
