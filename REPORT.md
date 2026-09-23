# REPORT: Python → Rust translation agent

| Trajectory | Model | `agent.py` | Model calls | Writes rejected by rule gate | Stop reason | Seed 0 |
|---|---|---|---|---|---|---|
| `run-20260923-133947` (**submitted `lib.rs`**) | claude-haiku-4-5 | final | 4 / 40 | 1 (+1 auto-rollback) | verified done | 100% |
| `run-20260923-133438` | gpt-5-nano | final | 5 / 40 | 3 | verified done | 100% |
| `run-20260923-124208` | claude-haiku-4-5 | earlier¹ | 8 / 40 | 2 | verified done | 100% |
| `run-20260923-124711` | gpt-5-nano | earlier¹ | 6 / 40 | 5 | verified done | 100% |

¹ Same design, before rejected drafts were carried over and before the Anthropic cache fix (§4).

## 1. Final score, and where it lost points

`python evaluate.py`: build PASS; `cargo test` 10/0; differential **2425/2425 (100%)**; semver.org chain PASS; `unsafe`, `.clone()`, `.unwrap()`, `todo!`, `panic!` and extra dependencies all **0**. It also scores 100% on unseen seeds 9999 and 31337 (n=300).

It can still lose points on the grading seed, and so would any correct translation. `evaluate.py`'s generator for *valid* versions sometimes emits strings the oracle rejects (e.g. `20.10.13-ly-o.05+x`, a numeric prerelease with a leading zero). Parse, compare, bump and round-trip on such inputs count as failures for every implementation. This `lib.rs` gets 99.59% on seed 4242 (n=300), which is that seed's ceiling. Accepting those strings would *gain* round-trip points, so we left it correct. One gap remains untested: numbers above `u64::MAX` return `Err` where Python accepts them, but the generator never produces them.

## 2. Agent vs. scaffold: roughly 40 / 60

The agent wrote every line of Rust, tests included. It also fixed its own bugs from failing-case feedback, such as a parser that took a `-` inside the build metadata as the prerelease separator.

The scaffold contributed the rest:

- **Prompt:** it chose the 173 relevant lines of Python (cut out with `ast`) and listed the known pitfalls. The first whole-file draft that passed the rule gate scored between 97.9% and 100% in every run, so most of the semantics arrived with the prompt.
- **Rule gate:** the *first* write of all four runs contained `.unwrap()`, `.expect(` or `panic!`, and the gate stopped each one.
- **Control loop:** verification, rollback and the decision to stop were all scaffold code.

This split is an estimate: we did not run an ablation without the pitfall list.

## 3. What the agent did that we did not intend

- **It routed around the rule.** Haiku's first draft used `.unwrap()` and `.expect(`. Told both were forbidden, it moved the same assertions into 15 × `panic!`. nano switched between the three for five drafts. `evaluate.py` counts `.unwrap()` but not `.expect(`, so an optimiser scored by `evaluate.py` alone would make exactly that swap. The gate had to name `expect`, `panic!` and `unreachable!` explicitly.
- **It misread its own state.** After its draft was rejected, the final Haiku run used `replace_item` to add just its `tests` module to the still-stub file, as if the draft had been applied. The score fell from −42.5 to −47.5, the change was rolled back, and the next call resent the full file.
- **It probed instead of fixing.** An earlier Haiku run spent three calls re-running one failing input. It then rewrote `parse` and reproduced the same bug (`s.find('-')` hit a hyphen in the build metadata).
- **It never used `note`, `finish`, `read_python` or `grep_python`.** Every run ended on verification before the agent claimed it was done.

## 4. Challenges and how we handled them

- **A reward signal that is sometimes unfixable and sometimes misleading** (§1). The agent's score leaves out cases whose input the oracle rejects. The agent is scored on a private fixed seed (4242) and must also pass two confirmation seeds. Nothing is tuned on seed 0.
- **Context.** The transcript is never replayed. Each call sends the static system prompt plus one snapshot: status, the agent's notes, a one-line-per-call log, the previous results, the latest verification and the current `lib.rs`. That kept every call between 16k and 32k characters. The first version dropped rejected drafts, so weak models regenerated about 400 lines, with new violations, every time. The draft is now carried over with the offending lines marked. Rejections went 2→1 (Haiku) and 5→3 (nano), but one run each proves little.
- **Termination.** "Done" means *verified*: 100% on the scoring and confirmation seeds, at least 8 passing tests and clean quality. The agent's own claim is not enough; nano reached 100% with 7 tests and was kept going for one more call. The run also stops on a stall (10 calls without a new best), a loop (the same action on the same file 3×) or silence (3 replies without a tool call). A regressing write is rolled back at once, and the best file is restored at the end.
- **Infrastructure.** One run crashed when the connection dropped mid-stream; the SDK only retries before a stream starts. `call_model` now retries transient errors itself. A retry is not counted as a model call because the failed attempt produced no reply; no retries happened in these runs. Anthropic's automatic cache breakpoint had landed on the changing snapshot, which paid for cache writes and never read one. It now sits on the system prompt. Cost: about $0.12 per Haiku run and $0.02 per nano run.
