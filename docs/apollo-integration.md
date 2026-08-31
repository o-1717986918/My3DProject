# Apollo runtime integration

## Source decision

Apollo is now the authoritative C++ competition runtime under
`runtime/apollo/`. It was imported from a fresh network clone of
<https://github.com/XiangruiJiang/ApolloCodebase>, not copied from the former
working-tree submodule. At the 2026-08-31 audit the public repository exposed
one branch, no tag/release/LFS/submodule, and the same head commit recorded in
`runtime/apollo/UPSTREAM.md`.

The online audit therefore found no hidden newer model or alternate branch.
The value of the migration is architectural reuse: Apollo already supplies a
complete C++ process lifecycle, behavior tree, world model, seven-role
assignment, formation logic, radio messages, A* walking, ONNX inference,
learned walking, learned recovery, and deployable packaging.

## Reused without reimplementation

- 78-to-23 self-contained walk network and its history/decoder;
- 75-to-23 learned get-up network;
- team-normalized world state and filtered ball estimate;
- dynamic GK/AP/ST/CBL/CBR/CDM/CBM role allocation;
- opponent-aware path planning and set-play behavior;
- compact inter-agent communication and source-complete GPL package flow.

## My3D migrations

- bounded `--max-cycles` execution and opt-in status telemetry;
- stable Python-era kick timing and distance/orientation gates, expressed as a
  C++ `KickCommand` and driven by Apollo's proven walk network;
- approach/kick/stabilize/cooldown integration inside the live behavior tree;
- defensive kickoff guard placement outside the inclusive goalkeeper area;
- strict 14-agent headless/visual acceptance and native CTest checks.

The 1200-cycle integration run observed 27 kick-state samples. A visible kick
raised measured forward ball velocity from approximately 0.04 m/s to 1.43 m/s,
while all 14 clients completed and the server logged zero illegal-defense foul.

## License boundary

Apollo declares GPL-3.0-or-later. The repository now carries that license at
the root, preserves upstream notices, and packages the corresponding modified
C++ source with the executable. ONNX Runtime is downloaded separately from its
official release and its license/notices remain in that distribution.

License-restricted training references are not copied into the runtime or
archive. This is also a technical safeguard: the current v4 residual actor is
not self-contained and cannot replace Apollo's walk model by filename alone.
