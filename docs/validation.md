# Validation record

Validation date: 2026-08-31
Platform: WSL2 Ubuntu 22.04
Client environment: `my3d-team`, Python 3.13.15
Server: RCSSServerMJ 0.2.1, `fifa7vs7`, `ssim26`, synchronous mode

## Apollo C++ release-path validation

The authoritative competition path was migrated to a fresh online import of
ApolloCodebase commit `71018c968969d6e55130b0e1987cd5b4f5c3b4df`.
The upstream branch/release/LFS/submodule audit and both ONNX asset digests are
recorded in `runtime/apollo/UPSTREAM.md`.

- CMake `RelWithDebInfo` build with GCC/C++17: passed.
- `runtime_config_test`: passed.
- `field_geometry_test`: passed.
- Strict 1200-cycle Apollo 7v7 acceptance: passed.
- Connections/joins/`PlayOn`/clean exits: 14/14/14/14.
- Fatal client failures/server errors/illegal defense: 0/0/0.
- Observed kick-state telemetry samples: 27.
- Observed visible kick: ball forward velocity increased from approximately
  0.04 m/s to 1.43 m/s during `KickForward`.
- RCSSServerMJ activation warnings: 14, isolated to the known add-player model
  recompilation event and reported separately.

The former 4.0 m defensive-kickoff baseline lay exactly on the server's
inclusive goalkeeper-area boundary and caused one illegal-defense penalty. It
is now 5.3 m from the goal line (1.3 m outside the area), protected by the
geometry regression test.

## Automated checks

- `pytest -q`: 40 passed (29 existing runtime checks plus 11 guarded
  reference-posture checks).
- `PYTHONPATH=training pytest training/tests -q`: 43 passed in `my3d-rl`.
- Client bytecode compilation: passed.
- Shell syntax validation: passed.
- Patch whitespace validation: passed.

The tests cover canonical field orientation, perception freshness, joint-state
mapping, finite/clamped motor output, beam lifecycle, attack phase transitions,
head tracking, player role assignment, asset integrity, the 80-to-23 model
boundary, posture entry guards, bounded blending, and same-cycle fallback.

## Single-player motion evidence

- The corrected joint sensor mapping and walking-policy lifecycle maintained an
  upright forward walk for approximately 25 m and reached a placed ball.
- Beam confirmation placed the attacker at its intended canonical formation
  coordinate near `(-9.5, -2.0)` before walking began.
- In a deterministic close-ball scenario, the stable kick implementation
  produced `KICK -> RECOVER -> APPROACH -> KICK`, remained upright, and moved
  the ball on successive attempts.
- A more aggressive hand-authored kick moved the ball but caused a repeatable
  fall, so it was removed from the competition path.
- The Apollo get-up adapter recovered from deterministic front, back, left, and
  right falls in the real simulator. Observed verification times were 2.48 s,
  1.92 s, 2.66 s, and 1.94 s respectively.

## Seven-versus-seven smoke test

The 2026-08-31 WSLg visual acceptance ran for 3000 bounded synchronous cycles
with the current stable competition policy. The window was closed after the
bounded run completed and the launcher cleaned up the server. Its final result
was:

```text
status=0 connected=14 play_on=14 failures=0 server_errors=0
```

During the visual run the attacker repeatedly traversed
`APPROACH -> ALIGN -> KICK -> RECOVER`, and both teams exercised the Apollo
get-up path. Host-local logs are under
`artifacts/visual-match-20260831-110630/`; generated match logs remain ignored
and are not part of the repository.

Two local seven-player teams completed an 800-cycle baseline run. After the
zone-owner and deterministic `PLAY_ON` changes, the final tree completed an
additional 500-cycle real-time bounded run. The current recovery and kickoff
tree then completed a 600-cycle acceptance run with the attack loop enforced:

- all 14 clients connected and received the correct active/passive side;
- both teams passed through their kickoff modes and reached `PLAY_ON`;
- the active #7 completed four full `ALIGN -> KICK -> RECOVER` cycles;
- five in-match falls were recovered by the Apollo policy;
- no placement foul was logged after deterministic left-team kickoff;
- no client traceback or logged client error occurred;
- all 14 clients exited cleanly at the configured cycle limit.

The repository-update release candidate repeated the full 600-cycle gate after
adding CI and the visual launcher. The observed result was:

```text
connected=14 play_on=14 shutdowns=14 failures=0
alignments=2 kicks=4 attack_recoveries=4 getups=6
```

The preserved evidence is `/tmp/my3d-match.lW7C8y` on the validated WSL host.
This path is host-local and is intentionally not committed.

The guarded reference-posture integration was then evaluated against an
800-cycle stable control and in three independent 800-cycle enabled matches. An
uncapped experiment was rejected after only 2/18 bursts completed and 16 hit
the posture guard. With the final 10% cap, the enabled matches completed 5/5,
5/5, and 16/16 bursts with zero posture/inference aborts. All had 14
connections, `PLAY_ON`, zero client failure, clean shutdown, and complete
`ALIGN -> KICK -> RECOVER` loops. Their observed get-up counts were 5, 8, and
3; the same-length stable control observed 5, so the integration is accepted
only as a bounded posture hint, not a standalone runner. One preserved local
record is `/tmp/my3d-match.Fu9mkj`; generated logs and restricted assets remain
uncommitted.

RCSSServerMJ printed one MuJoCo control warning during each player activation.
The warning coincided with the server's full-model recompilation on
`add_players`; it also occurred before meaningful policy control, while client
actions were independently finite and clamped. It did not prevent activation
or `PLAY_ON`. This is recorded as an upstream 0.2.1 activation risk, not hidden
as a passed client check.

## Repository quality gates

- Black formatting check: passed for runtime, scripts, tests, and training code.
- Flake8: passed with the repository's 100-column compatibility profile.
- GitHub Actions: runtime quality/tests on Python 3.13 and deployment-contract
  tests on Python 3.12.
- Workflow and project YAML parsing: passed.
- Credential-pattern and untracked-large-file review: no project secret or
  generated model selected for commit.

## Current competition readiness

The minimum match loop is operational: legal active/passive kickoff formation,
perception, canonical localisation, ball search, approach, alignment, repeated
stable ball contact, learned four-direction recovery, set-play response,
zone-based single-player ball ownership, forward support, and process shutdown.
The next performance work is stronger learned kicking, teammate communication,
collision avoidance, and opponent-aware role arbitration. These are
competitive-quality improvements rather than blockers for starting and
completing a 7v7 match.

The v4/GMR capability is present in the formal action stack behind explicit
activation and integrity checks. It is not release-default locomotion: stable
`walk.onnx` remains dominant/default, and full running promotion still requires
the R2/R3 command suite and three-seed ten-second acceptance.
