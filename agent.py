#!/usr/bin/env python3
"""Python -> Rust translation agent for reference/version.py.

    python agent.py                          # anthropic, claude-haiku-4-5
    python agent.py --provider openai        # openai, gpt-5-nano
    python agent.py --budget 40              # cap on model calls (graded: do not raise)

Design, one line per TODO:
  1. call_model()    Anthropic or OpenAI behind one provider-neutral message format.
  2. system_prompt() rules + required API + the relevant Python source (cut out
                     with `ast`) + the pitfalls first drafts get wrong.
  3. build_context() never replays the transcript. Every call is the static system
                     prompt plus ONE fresh snapshot: status, the agent's own notes
                     (WRITE), the current lib.rs and its latest verification
                     (SELECT), a one-line-per-call action log (COMPRESS), and only
                     the previous step's tool results in full.
  4. should_stop()   stops when the scaffold has *verified* done (the agent's claim
                     alone is not believed), when the best score stalls, when the
                     agent loops or stops calling tools. A write that lowers the
                     score is rolled back on the spot; the best version is kept.
  5. tools           every write is built, tested and scored automatically;
                     replace_item edits one item; run_cases probes the oracle.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, pathlib, re, subprocess, sys, tempfile, time

HERE   = pathlib.Path(__file__).parent
RUST   = HERE / "rust"
LIB    = RUST / "src" / "lib.rs"
BIN    = RUST / "target" / "release" / "harness"
PYSRC  = HERE / "reference" / "version.py"
LOGS   = HERE / "logs"

def _load_dotenv(path: pathlib.Path = HERE / ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency). Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if v and k not in os.environ:
            os.environ[k] = v

_load_dotenv()

# rustup only edits the interactive shell's rc; make cargo reachable anyway.
_CARGO_BIN = pathlib.Path.home() / ".cargo" / "bin"
if _CARGO_BIN.is_dir() and str(_CARGO_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = str(_CARGO_BIN) + os.pathsep + os.environ.get("PATH", "")

# Set from the command line in main().
CFG = {"provider": "anthropic", "model": "claude-haiku-4-5", "effort": "high", "budget": 40}
DEFAULT_MODEL = {"anthropic": "claude-haiku-4-5", "openai": "gpt-5-nano"}
FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")            # accept fallbacks="default"

# ============================================================== TODO 1
# Provider-neutral message format used by the loop and build_context():
#   {"role": "system" | "user", "content": str}
#   {"role": "assistant", "content": str,
#    "tool_calls": [{"id", "name", "arguments"}], "anthropic_blocks": [...]?}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}
# `anthropic_blocks` is the verbatim Anthropic content (incl. thinking blocks)
# and is replayed only when talking to Anthropic.

def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """Send `messages` + `tools` to a model; return its reply.

    Return shape expected by the loop below:
        {"text": str | None,
         "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...],
         "usage": dict, "stop_reason": str, "anthropic_blocks": list | None}

    Keys are read from the environment (or .env) - never hard-coded.
    A retry after a dropped connection is the same model call: it produced
    no reply, so it does not count against the budget.
    """
    fn = {"anthropic": _call_anthropic, "openai": _call_openai}.get(CFG["provider"])
    if fn is None:
        raise ValueError(f"unknown provider {CFG['provider']!r}")
    for attempt in range(1, 4):
        try:
            return fn(messages, tools)
        except Exception as e:
            if attempt == 3 or not _transient(e):
                raise
            print(f"      (transient {type(e).__name__}; retry {attempt} in {10 * attempt}s)")
            time.sleep(10 * attempt)

def _transient(e: Exception) -> bool:
    """Network drops, 429 and 5xx. The SDKs retry these only before a stream
    starts; a connection lost mid-stream surfaces as a raw httpx error."""
    import httpx
    if isinstance(e, httpx.TransportError):
        return True
    status = getattr(e, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(e).__name__ in ("APIConnectionError", "APITimeoutError")

_clients: dict = {}

def _call_anthropic(messages: list[dict], tools: list[dict]) -> dict:
    import anthropic
    client = _clients.get("anthropic") or _clients.setdefault(
        "anthropic", anthropic.Anthropic(max_retries=4))

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    conv: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            conv.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = m.get("anthropic_blocks")
            if not blocks:
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"],
                            "input": c.get("arguments") or {}}
                           for c in m.get("tool_calls") or []]
            conv.append({"role": "assistant",
                         "content": blocks or [{"type": "text", "text": "(no output)"}]})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m["content"] or "(no output)"}
            # every result of one assistant turn goes back in ONE user message
            prev = conv[-1] if conv else None
            if (prev and prev["role"] == "user" and isinstance(prev["content"], list)
                    and prev["content"] and prev["content"][0].get("type") == "tool_result"):
                prev["content"].append(block)
            else:
                conv.append({"role": "user", "content": [block]})

    kwargs = dict(
        model=CFG["model"],
        max_tokens=32000,
        # Cache only the stable prefix (tools + system). The snapshot changes
        # every call, so a breakpoint there would pay cache writes, never reads.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=conv,
        tools=[{"name": t["name"], "description": t["description"],
                "input_schema": t["parameters"]} for t in tools],
    )
    if CFG["model"].startswith("claude-haiku"):
        # Haiku 4.5 predates adaptive thinking and the effort parameter.
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
    else:
        kwargs.update(thinking={"type": "adaptive"}, output_config={"effort": CFG["effort"]})
    if CFG["model"].startswith(FALLBACK_MODELS):
        # Re-run a safety-declined request on Anthropic's recommended fallback.
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    # Streaming: a whole-file write plus thinking can outlast a plain request.
    with client.beta.messages.stream(**kwargs) as stream:
        resp = stream.get_final_message()

    text = "".join(b.text for b in resp.content if b.type == "text") or None
    calls = [{"id": b.id, "name": b.name, "arguments": b.input}
             for b in resp.content if b.type == "tool_use"]
    blocks = [b.to_dict() for b in resp.content]
    if resp.stop_reason == "refusal":
        text, calls, blocks = f"(model refused: {resp.stop_details})", [], None
    elif resp.stop_reason == "max_tokens":
        # a tool_use cut off mid-input is not safe to run or to replay
        note = "(output hit max_tokens; any tool call was truncated and dropped)"
        text, calls, blocks = f"{text or ''}\n{note}".strip(), [], None
    elif any(b["type"] not in ("text", "thinking", "redacted_thinking", "tool_use")
             for b in blocks):
        blocks = None                   # e.g. a fallback block: rebuild from text + tool_use
    u = resp.usage
    return {"text": text, "tool_calls": calls, "stop_reason": resp.stop_reason,
            "anthropic_blocks": blocks,
            "usage": {"input": u.input_tokens, "output": u.output_tokens,
                      "cache_read": u.cache_read_input_tokens or 0,
                      "cache_write": u.cache_creation_input_tokens or 0}}

def _call_openai(messages: list[dict], tools: list[dict]) -> dict:
    from openai import OpenAI
    client = _clients.get("openai") or _clients.setdefault("openai", OpenAI(max_retries=4))

    conv: list[dict] = []
    for m in messages:
        if m["role"] in ("system", "user"):
            conv.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c.get("arguments") or {})}}
                    for c in m["tool_calls"]]
            conv.append(msg)
        elif m["role"] == "tool":
            conv.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                         "content": m["content"] or "(no output)"})

    resp = client.chat.completions.create(
        model=CFG["model"], messages=conv,
        tools=[{"type": "function",
                "function": {"name": t["name"], "description": t["description"],
                             "parameters": t["parameters"]}} for t in tools])
    choice = resp.choices[0]
    calls = []
    for tc in choice.message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_invalid_json": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    u = resp.usage
    return {"text": choice.message.content, "tool_calls": calls,
            "stop_reason": choice.finish_reason, "anthropic_blocks": None,
            "usage": {"input": u.prompt_tokens, "output": u.completion_tokens} if u else {}}

# ============================================================== TODO 2
# Only these parts of version.py are graded; the rest (match, next_version,
# bump_prerelease, ...) is noise for this task and stays behind read_python.
_PY_METHODS = ("__init__", "_nat_cmp", "bump_major", "bump_minor", "bump_patch",
               "compare", "__str__", "parse")

def _python_excerpt() -> str:
    """The graded parts of version.py, cut out with `ast`, docstrings dropped."""
    if not PYSRC.exists():
        return "(reference/version.py missing - run fetch_source.py)"
    src = PYSRC.read_text()
    lines = src.splitlines()
    spans = []

    def add(node):
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        body = getattr(node, "body", None)
        doc = None
        if (isinstance(node, ast.FunctionDef) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            doc = (body[0].lineno, body[0].end_lineno)
        spans.append((start, node.end_lineno, doc))

    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_cmp":
            add(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Version":
            for item in node.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    tgt = item.target if isinstance(item, ast.AnnAssign) else item.targets[0]
                    if isinstance(tgt, ast.Name) and ("REGEX" in tgt.id or "IDENTIFIER" in tgt.id):
                        add(item)
                elif isinstance(item, ast.FunctionDef) and item.name in _PY_METHODS:
                    add(item)
    return "\n\n".join(
        "\n".join(lines[i - 1] for i in range(a, b + 1) if not (doc and doc[0] <= i <= doc[1]))
        for a, b, doc in spans)

_SYSTEM_PROMPT = """\
You are a Rust engineer translating a Python module into idiomatic, safe Rust. You act only through tools, inside an automated harness.

GOAL
Write rust/src/lib.rs: a faithful translation of the graded parts of python-semver's `version.py` (source below). It is graded by differential testing against the real Python `semver` package (thousands of generated parse / compare / bump / format cases), by the crate's own `cargo test`, and by code quality.

REQUIRED PUBLIC API (src/main.rs calls exactly these; keep them unchanged)
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Version { pub major: u64, pub minor: u64, pub patch: u64,
                         pub prerelease: Option<String>, pub build: Option<String> }
    pub fn parse(s: &str) -> Result<Version, String>
    pub fn to_string(v: &Version) -> String
    pub fn compare(a: &Version, b: &Version) -> std::cmp::Ordering
    pub fn bump_major(v: &Version) -> Version
    pub fn bump_minor(v: &Version) -> Version
    pub fn bump_patch(v: &Version) -> Version
You may add private helpers and a `#[cfg(test)] mod tests`.

HARD RULES (a write that breaks one is rejected before it touches the file)
- std only. No `unsafe`. No todo!/unimplemented!/panic!/unreachable!. No std::process (never call Python).
- No `.unwrap()` or `.expect(` anywhere, tests included. Use Result/Option combinators, `?`, match, if let.
- Avoid `.clone()`: create owned Strings where data is born (`s.to_string()`), borrow (&str, `as_deref()`) everywhere else.
- Nothing may panic: an out-of-range number in parse is an Err; bumps use saturating_add.

BEHAVIOUR - translate the Python exactly; do not "fix" it to match other semver libraries
- parse: the whole string must match the regex (re.ASCII; anchored ^...\\Z). No leading zeros in major/minor/patch or in numeric prerelease identifiers (build identifiers MAY have leading zeros). No empty identifiers. Identifiers use only [0-9A-Za-z-]. All three numbers are required; no "v" prefix. `_native_parse_parts` is an absent optional accelerator: ignore it.
- compare: major, minor, patch numerically; then prerelease via _nat_cmp. Build metadata is IGNORED. A version without prerelease is GREATER than the same version with one. Numeric identifiers compare numerically and rank LOWER than alphanumeric ones; alphanumeric ones compare in ASCII order; if all shared identifiers are equal, more identifiers is greater.
- bump_major / bump_minor / bump_patch: exactly the Python, e.g. bump_patch("1.2.3-rc.1+b") is "1.2.4" (prerelease and build dropped, patch still increments).
- to_string: "M.m.p", then "-<prerelease>" if present, then "+<build>" if present.

TESTS
Put at least 8 #[test] fns in `#[cfg(test)] mod tests`: valid and invalid parses, the semver.org chain
1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0,
build ignored in compare, bumps (incl. on prereleases), round-trips. Without unwrap, use helpers such as
    fn cmp(a: &str, b: &str) -> Result<Ordering, String> { Ok(compare(&parse(a)?, &parse(b)?)) }
    assert_eq!(cmp("1.0.0-alpha", "1.0.0"), Ok(Ordering::Less));
    assert!(parse("01.0.0").is_err());

HOW THIS HARNESS WORKS
- Every reply you send costs 1 of @@BUDGET@@ model calls. One reply may contain several tool calls; they run in order.
- You do not see your earlier messages. Each turn you receive a fresh snapshot: status, your notes, a compressed action log, the full results of your previous step, the latest verification report, and the current lib.rs. Use `note` for anything you need to remember.
- Every write (write_rust / replace_item) is automatically built, tested and scored against the Python oracle. The report is the tool result. If the score drops or the build fails, the harness restores the best version so far and shows you the failed attempt's report.
- Use run_cases to check specific inputs against the oracle.
- Be efficient: your FIRST reply should write the complete lib.rs with write_rust. Afterwards fix failures with replace_item (one fn at a time) or write_rust.
- The run ends by itself once verification passes (100% differential on the scoring seed and on fresh confirmation seeds, precedence chain, cargo test all passing, clean quality). Call finish only if you believe that is the case.

PYTHON SOURCE (graded parts of reference/version.py; read_python / grep_python show the rest)
```python
@@PYTHON@@
```
"""

def system_prompt() -> str:
    """Moderate baking: the rules and the known pitfalls are stated, the
    semantics come from the (pre-selected) source itself."""
    return (_SYSTEM_PROMPT.replace("@@BUDGET@@", str(CFG["budget"]))
                          .replace("@@PYTHON@@", _python_excerpt()))

# ============================================================ run state
# Shared by the tools (which verify and roll back), build_context (which shows
# it) and should_stop (which judges it).
class RunState:
    def __init__(self):
        self.step = 0
        self.best_src, self.best_score, self.best_step = "", float("-inf"), 0
        self.best_report, self.best_metrics = "", {}
        self.cur_score, self.cur_report, self.cur_metrics = None, "(not verified yet)", {}
        self.done = False
        self.notes = ""
        self.draft = ""             # last write rejected by the rule check, offending lines marked
        self.finish_rejected = self.no_call_streak = self.max_repeat = 0
        self.seen: dict[str, int] = {}

    def accept(self, src: str, score: float, report: str, metrics: dict) -> None:
        """`src` is now on disk and is at least as good as the best so far."""
        if score > self.best_score:
            self.best_step = self.step
        self.best_src, self.best_score = src, score
        self.best_report, self.best_metrics = report, metrics
        self.cur_score, self.cur_report, self.cur_metrics = score, report, metrics
        self.done = bool(metrics.get("done"))

STATE = RunState()

# ============================================================== TODO 3
CTX_TOOL_OUT  = 6000    # chars of each previous-step tool result shown in full
CTX_LOG_LINES = 30      # older calls, one line each

def _short_args(name: str, args: dict) -> str:
    if name == "write_rust":
        return f"<{len(args.get('content', ''))} chars>"
    if name == "replace_item":
        return f"name={args.get('name')!r}, <{len(args.get('code', ''))} chars>"
    if name == "note":
        return f"<{len(args.get('text', ''))} chars>"
    s = json.dumps(args)
    return s if len(s) <= 120 else s[:117] + "..."

def _steps(history: list[dict]) -> list[tuple[dict, list[dict], list[str]]]:
    """Group the transcript into (assistant turn, its tool results, harness nudges)."""
    steps: list = []
    for m in history[2:]:
        if m["role"] == "assistant":
            steps.append((m, [], []))
        elif steps and m["role"] == "tool":
            steps[-1][1].append(m)
        elif steps and m["role"] == "user":
            steps[-1][2].append(m["content"])
    return steps

def build_context(history: list[dict], step: int) -> list[dict]:
    """Turn the full history into the messages you actually send.

    The transcript is never replayed. Size is bounded no matter how many
    compiler errors pile up: static system prompt + one snapshot whose parts
    are each capped. (No earlier assistant turn is sent, so there are also no
    thinking blocks to keep byte-identical across calls.)
    """
    steps = _steps(history)
    log = []
    for i, (a, results, _) in enumerate(steps[:-1], 1):
        calls = a.get("tool_calls") or []
        acts = "; ".join(
            f"{c['name']}({_short_args(c['name'], c.get('arguments') or {})})"
            f" -> {(r['content'].splitlines() or [''])[0][:140]}"
            for c, r in zip(calls, results)) or "(no tool call)"
        said = " ".join((a.get("content") or "").split())
        log.append(f"call {i}: {acts}" + (f'  | said: "{said[:160]}"' if said else ""))

    prev = ["(this is your first call)"]
    if steps:
        a, results, nudges = steps[-1]
        prev = []
        if a.get("content"):
            prev.append(f"You said: {a['content'][:1500]}")
        for c, r in zip(a.get("tool_calls") or [], results):
            out = r["content"]
            if len(out) > CTX_TOOL_OUT:
                out = out[:CTX_TOOL_OUT] + f"\n... ({len(out) - CTX_TOOL_OUT} more chars cut)"
            prev.append(f"### {c['name']}({_short_args(c['name'], c.get('arguments') or {})})\n{out}")
        prev += [f"### harness\n{n}" for n in nudges]

    best = ("none yet" if STATE.best_score == float("-inf")
            else f"{STATE.best_score:.2f} (reached at call {STATE.best_step})")
    snapshot = "\n\n".join([
        f"TASK: {history[1]['content']}",
        "## Status\n"
        f"This is model call {step} of {CFG['budget']}. Best score so far: {best}. "
        f"Score = differential % - 50 per rule violation - 0.5 per failing test "
        f"- 0.1 per clone/unwrap/expect.",
        "## Your notes (edit with the `note` tool)\n" + (STATE.notes or "(empty)"),
        "## Action log (earlier calls, compressed)\n"
        + ("\n".join(log[-CTX_LOG_LINES:]) if log else "(none)"),
        "## Results of your previous call\n" + "\n\n".join(prev),
        "## Latest verification of the current lib.rs\n" + STATE.cur_report,
        "## Current rust/src/lib.rs\n```rust\n" + LIB.read_text() + "\n```",
    ] + ([
        # WRITE, not replay: without this the rejected code is gone and a weak
        # model regenerates the whole file, with a fresh set of violations.
        "## Your last REJECTED code (not applied)\nIt was rejected only for the lines marked "
        "`// <-- FORBIDDEN`. Fix exactly those and resend it with the same tool.\n"
        "```rust\n" + STATE.draft + "\n```",
    ] if STATE.draft else []) + [
        "Decide the next action and respond with tool calls.",
    ])
    return [history[0], {"role": "user", "content": snapshot}]

# ============================================================== TODO 4
PATIENCE = 10           # model calls without a new best score

def should_stop(history: list[dict], step: int, budget: int, last_score: float | None) -> tuple[bool, str]:
    """Return (stop?, why).

    "Done" is decided by verification, never by the agent's say-so: finish
    is only accepted if the checks agree. A lower score is handled where it
    happens - the write is rolled back - so stopping never leaves a worse
    lib.rs behind.
    """
    if STATE.done:
        return True, "done: verification passed (scoring seed + confirmation seeds, tests, quality)"
    if step >= budget:
        return True, f"budget exhausted ({budget} model calls)"
    if STATE.finish_rejected >= 2:
        return True, "gave up: agent claimed done twice, verification disagreed both times"
    if STATE.no_call_streak >= 3:
        return True, "stuck: 3 replies in a row without a tool call"
    if STATE.max_repeat >= 3:
        return True, "looping: the same action on the same lib.rs 3 times"
    if step - STATE.best_step >= PATIENCE:
        return True, f"stalled: best score {STATE.best_score:.2f} not improved in {PATIENCE} calls"
    return False, ""

# ============================================================ verification
SCORE_SEED, SCORE_N = 4242, 100          # fixed, so scores are comparable across calls
CONFIRM_RUNS = ((1, 300), (2, 300))      # fresh cases, only once the scoring seed is clean
MIN_TESTS, MAX_SMELLS = 8, 3

def _cargo(args: list[str], timeout: int = 300) -> tuple[bool, str]:
    try:
        p = subprocess.run(["cargo", *args], cwd=RUST, capture_output=True, text=True,
                           timeout=timeout, env={**os.environ, "CARGO_TERM_COLOR": "never"})
        return p.returncode == 0, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"

def _errors_only(out: str, limit: int = 4000) -> str:
    errs = [b for b in out.split("\n\n") if b.lstrip().startswith("error")]
    text = "\n\n".join(errs) if errs else out[-limit:]
    return text[:limit] + ("\n... (more errors cut)" if len(text) > limit else "")

def _test_failures() -> str:
    _, out = _cargo(["test", "--release"])
    if "failures:" in out:
        return out[out.index("failures:"):][:3000]
    return _errors_only(out, 3000)

def _ev():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import evaluate
    return evaluate

def _evaluate(seed: int, n: int) -> dict:
    """evaluate.py's own verdict: cargo test, quality, violations, precedence chain."""
    with tempfile.TemporaryDirectory() as d:
        js = pathlib.Path(d) / "r.json"
        try:
            p = subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--seed", str(seed),
                                "--n", str(n), "--json", str(js), "--quiet"],
                               cwd=HERE, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return {"error": "evaluate.py timed out (infinite loop in the harness?)"}
        return json.loads(js.read_text()) if js.exists() else {"error": (p.stdout + p.stderr)[-2000:]}

def _differential(seed: int, n: int) -> tuple[dict, list[str], str | None]:
    """evaluate.py's differential cases, minus the ones no translation can pass.

    evaluate.py's "valid" generator sometimes emits strings the oracle itself
    rejects (e.g. prerelease "05"). It then counts parse/compare/bump/format
    of them as failures for EVERY implementation - and a translation that
    wrongly accepts them even gains round-trip points. Showing those to the
    agent would send it chasing an unfixable (or harmful) change, so cases
    whose input the oracle rejects are skipped here.
    Returns ({family: (passed, total)}, first failures, error).
    """
    ev = _ev()
    valid, invalid, pairs, bumps = ev.build_cases(seed, n)
    cmds  = [f"parse {v}" for v in valid] + [f"parse {v}" for v in invalid]
    cmds += [f"compare {x} {y}" for x, y in pairs]
    cmds += [f"bump {k} {v}" for k, v in bumps] + [f"format {v}" for v in valid]
    got, err = ev.run_harness(cmds)
    if err:
        return {}, [], err
    good = lambda s: ev.ref_parse(s) is not None
    fields = ("major", "minor", "patch", "prerelease", "build")

    def chk_valid(s, g):
        if not good(s): return None
        want = ev.ref_parse(s)
        if not g.get("ok"): return False, f"parse({s!r}) rejected; expected {want}"
        mine = {f: g.get(f) for f in fields}
        return mine == want, f"parse({s!r}) -> {mine}, expected {want}"
    def chk_invalid(s, g):
        if good(s): return None
        return not g.get("ok"), f"parse({s!r}) accepted, but it is invalid"
    def chk_cmp(xy, g):
        x, y = xy
        if not (good(x) and good(y)): return None
        want = ev.ref_compare(x, y)
        return g.get("ok") and g.get("cmp") == want, f"compare({x!r},{y!r}) -> {g.get('cmp')}, expected {want}"
    def chk_bump(kv, g):
        kind, s = kv
        if not good(s): return None
        want = ev.ref_bump(kind, s)
        return g.get("ok") and g.get("version") == want, f"bump {kind} {s!r} -> {g.get('version')!r}, expected {want!r}"
    def chk_round(s, g):
        if not good(s): return None
        out = g.get("version")
        return g.get("ok") and ev.ref_parse(out) == ev.ref_parse(s), f"format({s!r}) -> {out!r}, not equivalent"

    fam, fails, i = {}, [], 0
    for name, items, check in (("parse valid", valid, chk_valid), ("parse invalid", invalid, chk_invalid),
                               ("compare", pairs, chk_cmp), ("bump", bumps, chk_bump),
                               ("round-trip", valid, chk_round)):
        passed = total = 0
        for k, item in enumerate(items):
            try:
                r = check(item, got[i + k])
            except Exception as e:
                r = (False, f"{name} {item!r}: checker error: {e}")
            if r is None:
                continue
            total += 1
            passed += bool(r[0])
            if not r[0] and len(fails) < 12:
                fails.append(f"{name}: {r[1]}")
        fam[name] = (passed, total)
        i += len(items)
    return fam, fails, None

def _pct(fam: dict) -> float:
    total = sum(t for _, t in fam.values())
    return 100 * sum(p for p, _ in fam.values()) / total if total else 0.0

def verify() -> tuple[float, dict, str]:
    """Build, test and score the current lib.rs. Returns (score, metrics, report)."""
    ok, out = _cargo(["build", "--release"])
    if not ok:
        return -100.0, {"unmet": ["build fails"]}, "BUILD FAILED\n" + _errors_only(out)
    R = _evaluate(SCORE_SEED, SCORE_N)
    fam, fails, err = _differential(SCORE_SEED, SCORE_N)
    if err or "quality" not in R:
        return -90.0, {"unmet": ["harness aborted"]}, f"HARNESS ABORTED: {err or R.get('error')}"

    diff, chain, viol = _pct(fam), R["spec_precedence_chain"], R["violations"]
    tests = R.get("cargo_test") or {"passed": 0, "failed": 10}     # None: tests don't compile
    q = R["quality"]
    expects = len(re.findall(r"\.expect\(", re.sub(r"//.*", "", LIB.read_text())))
    smells = q["clone_calls"] + q["unwrap_calls"] + expects
    score = diff - 50 * len(viol) - 0.5 * min(tests["failed"], 10) - 0.1 * smells

    lines = [
        f"build OK | differential {diff:.2f}% (seed {SCORE_SEED}) | "
        f"precedence chain {'PASS' if chain else 'FAIL'} | score {score:.2f}",
        "families: " + "  ".join(f"{k} {p}/{t}" for k, (p, t) in fam.items())
        + f"   (evaluate.py raw: {R.get('differential_pct', 0):.2f}%, it also counts inputs "
          f"the oracle rejects)",
        f"cargo test: {tests['passed']} passed, {tests['failed']} failed"
        + ("" if R.get("cargo_test") else " (tests did not run - do they compile?)"),
        f"quality: clone {q['clone_calls']}, unwrap {q['unwrap_calls']}, expect {expects}, "
        f"violations {', '.join(viol) or 'none'}",
    ]
    if fails:
        lines += ["failing cases (first 12):"] + [f"  - {f}" for f in fails]
    if tests["failed"] or not R.get("cargo_test"):
        lines += ["cargo test output:", _test_failures()]

    unmet = []
    if diff < 100:  unmet.append(f"differential {diff:.2f}% < 100%")
    if not chain:   unmet.append("precedence chain fails")
    if viol:        unmet.append(f"rule violations: {', '.join(viol)}")
    if tests["failed"] or tests["passed"] < MIN_TESTS:
        unmet.append(f"cargo test needs >= {MIN_TESTS} passing and 0 failing")
    if smells > MAX_SMELLS:
        unmet.append(f"{smells} clone/unwrap/expect calls (max {MAX_SMELLS})")
    if not unmet:
        for seed, n in CONFIRM_RUNS:
            fam2, f2, err2 = _differential(seed, n)
            d2 = 0.0 if err2 else _pct(fam2)
            lines.append(f"confirmation seed {seed} (n={n}): {d2:.2f}%" + (f" ({err2})" if err2 else ""))
            if d2 < 100:
                unmet.append(f"confirmation seed {seed}: {d2:.2f}% < 100%")
                lines += [f"  - {f}" for f in f2]
                break
    lines.append("DONE: every check passes" if not unmet else "NOT DONE: " + "; ".join(unmet))
    return score, {"unmet": unmet, "done": not unmet}, "\n".join(lines)

# ================================================================== tools
_FORBIDDEN = [(r"\bunsafe\b", "unsafe"), (r"\btodo!", "todo!"),
              (r"\bunimplemented!", "unimplemented!"), (r"\bpanic!", "panic!"),
              (r"\bunreachable!", "unreachable!"), (r"std::process|\bCommand::new", "std::process"),
              (r"\.unwrap\(\)", ".unwrap()"), (r"\.expect\(", ".expect(")]

def _forbidden(code: str) -> list[str]:
    """Offending lines of `code`, e.g. ["line 12 (.unwrap()): let x = y.unwrap();"]."""
    hits = []
    for n, line in enumerate(code.splitlines(), 1):
        nodoc = re.sub(r"//.*", "", line)    # same comment rule as evaluate.py
        hits += [f"line {n} ({label}): {line.strip()[:100]}"
                 for pat, label in _FORBIDDEN if re.search(pat, nodoc)]
    return hits

def _apply_source(new_src: str, checked: str, how: str) -> str:
    """Write, verify, and keep the result only if it is not worse than the best."""
    bad = _forbidden(checked)
    if bad:
        shown = "\n".join(f"  {b}" for b in bad[:15])
        more = f"\n  ... and {len(bad) - 15} more" if len(bad) > 15 else ""
        marked = [line + "  // <-- FORBIDDEN" if _forbidden(line) else line
                  for line in checked.splitlines()]
        STATE.draft = "\n".join(marked)
        return (f"REJECTED, file unchanged: {len(bad)} forbidden construct(s) (see HARD RULES). "
                f"Your code is kept in the next snapshot with these lines marked:\n{shown}{more}")
    STATE.draft = ""
    LIB.write_text(new_src)
    score, metrics, report = verify()
    if score < STATE.best_score:
        LIB.write_text(STATE.best_src)
        _cargo(["build", "--release"])       # keep the harness binary in sync for run_cases
        STATE.cur_score, STATE.cur_report = STATE.best_score, STATE.best_report
        STATE.cur_metrics = STATE.best_metrics
        return (f"REVERTED: {how}, but score {score:.2f} < best {STATE.best_score:.2f}; "
                f"lib.rs was restored to the best version.\nReport of the rejected attempt:\n{report}")
    tag = "new best" if score > STATE.best_score else "= best"
    STATE.accept(new_src, score, report, metrics)
    return f"{how}; score {score:.2f} ({tag})\n{report}"

def _skip_literal(src: str, i: int) -> int:
    """If a comment/string/char literal starts at i, return the index after it, else i."""
    if src.startswith("//", i):
        j = src.find("\n", i)
        return len(src) if j < 0 else j
    if src.startswith("/*", i):
        j = src.find("*/", i + 2)
        return len(src) if j < 0 else j + 2
    if src[i] == '"':
        j = i + 1
        while j < len(src) and src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        return j + 1
    if src[i] == "'":
        m = re.match(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])'", src[i:])
        return i + m.end() if m else i + 1       # no match: a lifetime
    return i

def _find_item(src: str, name: str) -> tuple[int, int] | None:
    """Span of a top-level fn/struct/enum/mod/... called `name`, with its doc comments and attributes."""
    m = re.search(rf"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?"
                  rf"(?:fn|struct|enum|mod|trait|type|const|static)\s+{re.escape(name)}\b", src, re.M)
    if not m:
        return None
    start = m.start()
    while True:                              # absorb doc comments / attributes above it
        prev_end = src.rfind("\n", 0, start - 1) if start > 0 else -1
        prev_line = src[prev_end + 1:start].strip()
        if start > 0 and (prev_line.startswith("///") or prev_line.startswith("#[")):
            start = prev_end + 1
        else:
            break
    i, depth = m.end(), 0
    while i < len(src):
        j = _skip_literal(src, i)
        if j != i:
            i = j
            continue
        c = src[i]
        if c == ";" and depth == 0:
            break
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    else:
        return None
    end = i + 1
    if src[end:end + 1] == "\n":
        end += 1
    return start, end

def t_read_python(args):
    """The source to translate, by line range."""
    if not PYSRC.exists():
        return "reference/version.py missing - run fetch_source.py"
    lines = PYSRC.read_text().splitlines()
    a = max(1, int(args.get("start") or 1))
    b = min(len(lines), int(args.get("end") or a + 199), a + 299)
    return (f"reference/version.py lines {a}-{b} of {len(lines)}\n"
            + "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(a, b + 1)))

def t_grep_python(args):
    try:
        rx = re.compile(args.get("pattern", ""))
    except re.error as e:
        return f"bad regex: {e}"
    lines = PYSRC.read_text().splitlines()
    hits = [i for i, l in enumerate(lines, 1) if rx.search(l)]
    return "\n".join([f"{len(hits)} matching lines in reference/version.py"]
                     + [f"{i:4d}  {lines[i - 1]}" for i in hits[:40]])

def t_write_rust(args):
    """Overwrite rust/src/lib.rs. `content` must be the WHOLE file."""
    content = args["content"]
    return _apply_source(content, content, f"wrote lib.rs ({len(content)} chars)")

def t_replace_item(args):
    name, code = args["name"], args["code"].strip("\n") + "\n"
    src = LIB.read_text()
    span = _find_item(src, name)
    if span is None:
        return _apply_source(src.rstrip("\n") + "\n\n" + code, code,
                             f"'{name}' not found, appended it at the end")
    a, b = span
    return _apply_source(src[:a] + code + src[b:], code, f"replaced '{name}'")

def _oracle(cmd: str, got: dict):
    """(what the Python oracle says, whether the Rust answer matches)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import evaluate as ev
    p = cmd.split()
    fields = ("major", "minor", "patch", "prerelease", "build")
    try:
        if p[0] == "parse" and len(p) == 2:
            want = ev.ref_parse(p[1])
            mine = {f: got.get(f) for f in fields} if got.get("ok") else None
            return (want if want is not None else "invalid"), mine == want
        if p[0] == "compare" and len(p) == 3:
            if ev.ref_parse(p[1]) is None or ev.ref_parse(p[2]) is None:
                return "invalid operand", not got.get("ok")
            want = ev.ref_compare(p[1], p[2])
            return want, bool(got.get("ok")) and got.get("cmp") == want
        if p[0] == "bump" and len(p) == 3 and p[1] in ("major", "minor", "patch"):
            if ev.ref_parse(p[2]) is None:
                return "invalid", not got.get("ok")
            want = ev.ref_bump(p[1], p[2])
            return want, bool(got.get("ok")) and got.get("version") == want
        if p[0] == "format" and len(p) == 2:
            if ev.ref_parse(p[1]) is None:
                return "invalid", not got.get("ok")
            want = str(ev.ref.Version.parse(p[1]))
            return want, bool(got.get("ok")) and got.get("version") == want
    except Exception as e:
        return f"oracle error: {e}", False
    return "unknown command (use parse/compare/bump/format)", False

def t_run_cases(args):
    cmds = [c.strip() for c in args.get("commands") or [] if isinstance(c, str) and c.strip()][:20]
    if not cmds:
        return "no commands given"
    if not BIN.exists():
        return "harness binary missing - the crate has not built yet"
    try:
        p = subprocess.run([str(BIN)], input="\n".join(cmds) + "\n",
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "harness timed out (infinite loop?)"
    outs = [l for l in p.stdout.splitlines() if l.strip()]
    rows, bad = [], 0
    for i, cmd in enumerate(cmds):
        line = outs[i] if i < len(outs) else '{"ok":false,"error":"no output"}'
        try:
            got = json.loads(line)
        except json.JSONDecodeError:
            got = {"ok": False, "error": "unparseable"}
        want, match = _oracle(cmd, got)
        bad += not match
        rows.append(f"{'ok ' if match else 'BAD'} {cmd}\n    rust:   {line}\n    oracle: {want}")
    return f"{len(cmds)} cases: {len(cmds) - bad} match, {bad} MISMATCH\n" + "\n".join(rows)

def t_note(args):
    STATE.notes = str(args.get("text", ""))[:2000]
    return f"notes saved ({len(STATE.notes)} chars); they appear in every future snapshot"

def t_finish(args):
    if STATE.done:
        return "ACCEPTED: verification agrees the translation is done."
    STATE.finish_rejected += 1
    unmet = STATE.cur_metrics.get("unmet") or ["not verified yet"]
    return "REJECTED: verification says not done: " + "; ".join(unmet)

# Coarse enough to be cheap (one write = build + test + score), fine enough to
# not lose work (replace_item touches one item). No separate build/test/eval
# tools: every write already returns that report.
_NO_ARGS = {"type": "object", "properties": {}}
TOOLS = [
    dict(name="write_rust",
         description="Overwrite rust/src/lib.rs with the COMPLETE file. It is then built, tested "
                     "and scored automatically; the report is returned. Rolled back if the score drops.",
         parameters={"type": "object", "required": ["content"],
                     "properties": {"content": {"type": "string"}}}, fn=t_write_rust),
    dict(name="replace_item",
         description="Replace one top-level item (fn, struct, enum, mod, const...) of lib.rs, found by "
                     "name, with `code` (the complete new item, incl. doc comments/attributes; use "
                     "name 'tests' for the test module). Appends it if no item has that name. "
                     "Then built, tested, scored and possibly rolled back like write_rust.",
         parameters={"type": "object", "required": ["name", "code"],
                     "properties": {"name": {"type": "string"}, "code": {"type": "string"}}},
         fn=t_replace_item),
    dict(name="run_cases",
         description="Run up to 20 harness commands through the current build and compare each with "
                     "the Python oracle. Commands: 'parse <v>', 'compare <a> <b>', "
                     "'bump <major|minor|patch> <v>', 'format <v>'.",
         parameters={"type": "object", "required": ["commands"],
                     "properties": {"commands": {"type": "array", "items": {"type": "string"}}}},
         fn=t_run_cases),
    dict(name="read_python",
         description="Read lines of reference/version.py (at most 300 per call).",
         parameters={"type": "object", "properties": {"start": {"type": "integer"},
                                                      "end": {"type": "integer"}}},
         fn=t_read_python),
    dict(name="grep_python",
         description="Regex search over reference/version.py; returns matching lines with numbers.",
         parameters={"type": "object", "required": ["pattern"],
                     "properties": {"pattern": {"type": "string"}}}, fn=t_grep_python),
    dict(name="note",
         description="Replace your persistent notes (max 2000 chars), shown in every future snapshot.",
         parameters={"type": "object", "required": ["text"],
                     "properties": {"text": {"type": "string"}}}, fn=t_note),
    dict(name="finish",
         description="Claim the translation is complete. Only accepted if verification agrees.",
         parameters={"type": "object", "properties": {"summary": {"type": "string"}}}, fn=t_finish),
]
BY_NAME = {t["name"]: t for t in TOOLS}
SCHEMAS = [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]

# =================================================================== loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40, help="max model calls (graded cap: 40)")
    ap.add_argument("--task", default="Translate reference/version.py into rust/src/lib.rs.")
    ap.add_argument("--provider", choices=("anthropic", "openai"), default=CFG["provider"])
    ap.add_argument("--model", help="default: claude-haiku-4-5 / gpt-5-nano")
    ap.add_argument("--effort", default=CFG["effort"],
                    choices=("low", "medium", "high", "xhigh", "max"),
                    help="Anthropic output_config.effort (ignored by Haiku 4.5)")
    a = ap.parse_args()
    CFG.update(provider=a.provider, effort=a.effort, budget=a.budget,
               model=a.model or DEFAULT_MODEL[a.provider])

    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    def rec(**kw):
        with log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    history = [{"role": "system", "content": system_prompt()},
               {"role": "user",   "content": a.task}]
    rec(event="start", budget=a.budget, task=a.task,
        provider=CFG["provider"], model=CFG["model"], effort=CFG["effort"])

    # Baseline: whatever lib.rs holds now is the version to beat (not a model call).
    score, metrics, report = verify()
    STATE.accept(LIB.read_text(), score, report, metrics)
    print(f"[0] baseline score {score:.2f}")
    rec(event="baseline", score=score, report=report)

    step = 0
    while True:
        stop, why = should_stop(history, step, a.budget, STATE.cur_score)
        if stop:
            print(f"\n[stop] {why}")
            rec(event="stop", reason=why, steps=step, best=STATE.best_score)
            break

        step += 1
        STATE.step = step
        ctx = build_context(history, step)
        rec(event="context", step=step, chars=sum(len(m["content"]) for m in ctx))
        reply = call_model(ctx, SCHEMAS)
        # thinking signatures are large and useless in the trajectory log
        rec(event="model", step=step,
            reply={k: v for k, v in reply.items() if k != "anthropic_blocks"})

        if reply.get("text"):
            print(f"[{step}] {reply['text'][:200]}")
        history.append({"role": "assistant", "content": reply.get("text") or "",
                        "tool_calls": reply.get("tool_calls", []),
                        "anthropic_blocks": reply.get("anthropic_blocks")})

        calls = reply.get("tool_calls") or []
        if not calls:
            # Not done (only verification decides that) - stuck or chatting.
            STATE.no_call_streak += 1
            history.append({"role": "user", "content":
                            "No tool call received. Every reply must call at least one tool."})
            print(f"[{step}] (no tool call)")
            continue
        STATE.no_call_streak = 0

        for c in calls:
            tool = BY_NAME.get(c["name"])
            args = c.get("arguments") or {}
            key = hashlib.sha1((c["name"] + json.dumps(args, sort_keys=True)
                                + LIB.read_text()).encode()).hexdigest()
            STATE.seen[key] = STATE.seen.get(key, 0) + 1
            STATE.max_repeat = max(STATE.max_repeat, STATE.seen[key])
            if tool is None:
                out = f"unknown tool {c['name']!r}"
            elif "_invalid_json" in args:
                out = f"arguments were not valid JSON: {args['_invalid_json'][:200]!r}"
            else:
                try:
                    out = tool["fn"](args)
                except Exception as e:          # a bad call is feedback, not a crash
                    out = f"tool error: {type(e).__name__}: {e}"
            print(f"[{step}]   -> {c['name']}: {str(out).splitlines()[0][:120] if out else ''}")
            rec(event="tool", step=step, name=c["name"], output=str(out)[:4000])
            history.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                            "content": str(out)})
        rec(event="state", step=step, score=STATE.cur_score, best=STATE.best_score,
            best_step=STATE.best_step, usage=reply.get("usage"))

    if LIB.read_text() != STATE.best_src:        # never end on a worse file
        LIB.write_text(STATE.best_src)
        print(f"[end] restored best version (score {STATE.best_score:.2f})")
    print(f"\ntrajectory: {log}")
    print("final score:")
    subprocess.run([sys.executable, str(HERE / "evaluate.py")], cwd=HERE)

if __name__ == "__main__":
    main()
