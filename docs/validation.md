# Validation record

Validation date: 2026-08-30
Platform: WSL2 Ubuntu 22.04
Client environment: `my3d-team`, Python 3.13.15
Server: RCSSServerMJ 0.2.1, `fifa7vs7`, `ssim26`, synchronous mode

## Automated checks

- `pytest -q`: 29 passed.
- Client bytecode compilation: passed.
- Shell syntax validation: passed.
- Patch whitespace validation: passed.

The tests cover canonical field orientation, perception freshness, joint-state
mapping, finite/clamped motor output, beam lifecycle, attack phase transitions,
head tracking, and player role assignment.

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
