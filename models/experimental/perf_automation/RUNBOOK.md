# Runbook — optimize ANY model pipeline with perf_automation

A copy-pasteable, gotcha-aware guide for a newcomer **or an agent** to point the
perf_automation loop at a brand-new model and get a real, measured optimization
pass. Read `README.md` for the *design*; this file is the *operational* recipe.

The loop: profile a baseline → route to the slowest op bucket → an LLM picks &
applies one change → gate correctness (PCC) on hardware → re-measure → keep it
only if it's genuinely faster, else revert → repeat until target/budget/max-iter.

---

## 0. The 6 gotchas that will bite you (read first)

These cost real time on the first bring-up. Every one has a fix below.

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pytest` / `torch` during a stage | the tool shells out to bare `python`; it resolved to the wrong venv | run with, and put first on `PATH`, the **one** interpreter that has `torch`+`ttnn`+`pytest`+`claude-agent-sdk`+`tt-perf-report` (§1) |
| `TT_FATAL … Custom fabric mesh graph descriptor path must be specified` at device open | you isolated a **single chip of a multi-chip board** (e.g. p300 = 2 chips) | expose a **complete board**: `--devices 0,1` (not `--devices single`) (§3) |
| `Profiler DRAM buffers were full, markers were dropped` → tracy asserts `Op <id> not present in cpp_device_perf_report.csv` | model is op-dense; one capture exceeds the 12000-marker/RISC buffer | profile a **bounded, self-flushing** workload (§2) |
| Discovery picks the wrong/heavy test, or emits a glob PCC path that fails validation | the discovery sub-agent is non-deterministic | **pin** `--perf-test … -k <case>` (§4) |
| `revert skipped … pathspec did not match` and a discarded edit stays on disk | the model dir is **untracked**, so git REVERT has no baseline | give the model a **local git baseline** (§5) |
| Two jobs fight over the device / hang / OOM | another device-heavy job is running on the same chips | run on a **free board** or wait; never run two device jobs on the same chips |

---

## 1. One-time setup

**Interpreter.** The tool runs `python -m pytest` / `python -m tracy` as subprocesses
via bare `python`, so the *first* `python` on `PATH` must be the venv that has
everything. On a tt-metal checkout that's the build venv (here `python_env`), NOT
a side venv. Install the agent deps into THAT venv:

```bash
TTM=/home/ttuser/tt-metal          # adjust to your checkout
PY=$TTM/python_env/bin/python      # the venv with torch+ttnn+pytest
$PY -m pip install claude-agent-sdk==0.2.95 tt-perf-report==1.2.2 python-dotenv
$PY - <<'EOF'                       # sanity: all five import in ONE interpreter
import torch, ttnn, claude_agent_sdk, tt_perf_report, dotenv; print("deps ok")
EOF
which claude                        # the Agent SDK spawns this CLI; must be on PATH
```

Always export this before any run:

```bash
export TT_METAL_HOME=$TTM PYTHONPATH=$TTM
export PATH=$TTM/python_env/bin:$HOME/.local/bin:$PATH   # venv FIRST, then claude CLI
```

**Credentials.** The loop reads LLM creds ONLY from `perf_automation/.env.agent`
(never the shell env). Create it (keep it `chmod 600`, never commit it):

```
LITELLM_BASE_URL=https://<your-litellm-proxy>
LITELLM_API_KEY=sk-...
AGENT_MODEL_LEAD=anthropic/claude-sonnet-4-6      # SELECT / PLAN reasoning
AGENT_MODEL_SUB=anthropic/claude-haiku-4-5-20251001
AGENT_MODEL_EDIT=anthropic/claude-sonnet-4-6      # the editor — see note below
```

> The editor defaults to the cheap SUB (haiku) tier. On a complex model the haiku
> editor tends to emit edits that pass PCC but don't change the executed device ops
> (zero speedup). If laps keep showing `delta 0.0`, set `AGENT_MODEL_EDIT` to sonnet.

Verify it loads:

```bash
cd $TTM/models/experimental/perf_automation
$PY -c "from agent import config; c=config.load_agent_env('.env.agent'); \
print('base',bool(c['LITELLM_BASE_URL']),'key',bool(c['LITELLM_API_KEY'])); \
print({r:config.get_model(r,c) for r in ('lead','sub','edit')})"
```

---

## 2. Pick a perf workload (the #1 thing newcomers get wrong)

The loop needs two things from your model: a **correctness gate** (a PCC test) and
a **perf workload** to profile under Tracy. They can be the same test or different.

**If your model is small / single-forward:** the existing e2e test is fine; skip to §3.

**If your model is autoregressive or op-dense** (LLMs, TTS, anything that loops a
decoder for many tokens): a full e2e run emits *millions* of op markers and
**overflows the on-device profiler buffer** (12000 markers/RISC) — markers get
dropped and tracy post-processing crashes. Don't fight it; profile a **bounded,
self-flushing** slice that exercises the same op mix. Add a tiny perf test:

```python
# tests/.../test_<model>_perf.py  — a PROFILING workload, not a correctness gate
import pytest, torch, ttnn
from <your model> import pipeline   # reuse the real pipeline, don't fork it

PERF_MAX_NEW_TOKENS = 4   # few enough to stay under the marker buffer; tune down if needed

@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("n", [PERF_MAX_NEW_TOKENS], ids=["in0"])
def test_perf(device_params, device, n):
    # Flush the device profiler at natural boundaries so accumulated markers
    # never exceed the per-RISC buffer. Use whatever per-step hook your pipeline
    # exposes (here: on_encode / on_step callbacks). pipeline source stays untouched.
    flush = lambda *a: ttnn.ReadDeviceProfiler(device)
    result = pipeline.run(device, ..., max_new_tokens=n, on_encode=flush, on_step=flush)
    assert result            # sanity only; correctness lives in the PCC gate
```

Validate the workload profiles cleanly BEFORE wiring it in (no dropped markers,
an `ops_perf_results_*.csv` is produced):

```bash
TT_VISIBLE_DEVICES=0,1 TT_METAL_VISIBLE_DEVICES=0,1 TT_METAL_DEVICE_PROFILER=1 \
  $PY -m tracy -v -r -p -o /tmp/probe -m pytest <perf_test>::test_perf -k in0 -sv \
  2>&1 | grep -E "markers were dropped|OPs csv generated|passed|FAILED"
```

If you still see "markers were dropped", lower `PERF_MAX_NEW_TOKENS` (or flush more
often). `TT_METAL_PROFILER_MID_RUN_DUMP=1` alone is **not** enough for very dense models.

---

## 3. Device selection (multi-chip boards)

A p300 is a **2-chip board**. Exposing a single chip (`--devices single` →
`TT_VISIBLE_DEVICES=0`) makes tt-metal see "1 of 2 chips" → CUSTOM cluster → fatal.
Always expose a **complete board**:

```bash
# board 0 = chips 0,1   |   board 1 = chips 2,3
--devices 0,1     # model still runs on device 0; the 2nd chip just completes the board
```

If another device-heavy job is already running, put yours on the **other** board
(`--devices 2,3`) — and never share chips between two jobs.

---

## 4. Baseline (Before Loop)

Pin the perf test and case so you don't depend on the flaky discovery sub-agent:

```bash
cd $TTM/models/experimental/perf_automation
$PY -m agent.before_loop \
    $TTM/models/demos/<your_model> \
    --metric device_ms --devices 0,1 \
    --perf-test models/demos/<your_model>/tests/.../test_perf.py::test_perf -k in0
```

Success looks like `✔ tracy_baseline: device <N> ms · … buckets` and a printed
bucket table (matmul / datamove / reduction / …). That's your routing map.
Artifacts land in `runs/<id>/` (and `runs/latest` points at it).

---

## 5. Make REVERT/COMMIT real (recommended for multi-lap runs)

The loop's COMMIT (persist a win) and REVERT (roll back a discard) are **git-scoped
to the model dir**. If the model dir is **untracked**, REVERT can't restore — a
discarded edit is logged ("edit left on disk") and stays. For a clean multi-lap run,
give the model a **local-only** git baseline first:

```bash
cd $TTM
# exclude scratch so the commit is small (profiler captures, attempt logs, backups…)
cat >> models/demos/<your_model>/.gitignore <<'EOF'
_captured/
__pycache__/
*.pyc
EOF
git add -- models/demos/<your_model>
git commit -m "[local-only · DO NOT PUSH] <model> baseline for perf_automation"
```

Now REVERT does `git checkout <clean-sha> -- <model dir>` (scoped — your unrelated
work is never touched) and COMMIT persists kept wins as small incremental commits.

---

## 6. Run the optimization loop

```bash
cd $TTM/models/experimental/perf_automation
$PY -m agent.loop --until DECIDE   # ONE full lap (profiles, edits, gates, measures) — start here
$PY -m agent.loop                  # FULL session: laps until target / budget / max-iter
```

**Who decides the laps?** The tool, via `exit_policy.check_exit()` each lap, in order:
target metric reached → DONE; cost ≥ budget → STOPPED; iteration ≥ max-iter →
STOPPED; no untried levers → STOPPED; else continue. `--until <STATE>` is a *manual*
override that halts the engine at that state (e.g. `--until DECIDE` = one lap).

**Read the result** in `runs/latest/`:
- `ledger.jsonl` — one row per lap: lever, before/after, `delta`, `pcc`, kept/discarded + why.
- `state.json` — current metric (`baseline` / `current` / `target`) and counters.
- `events.jsonl` — the execution trace (every stage start/done, incl. COMMIT/REVERT).

**Healthy vs stuck:** a kept lap shows a negative `delta` (for `device_ms`) and
`result: keep`. If every lap is `delta: 0.0 … no_gain`, the edits aren't biting —
stop and fix targeting/edit-model (§1 note) rather than burn budget.

---

## 7. Agent checklist (TL;DR for an automated operator)

1. Deps in the torch+ttnn venv; that venv first on `PATH`; `.env.agent` present. (§1)
2. Identify the PCC gate + a perf workload; if autoregressive/op-dense, add a bounded
   self-flushing perf test and confirm "no markers dropped + OPs csv generated". (§2)
3. Choose a complete, **free** board (`--devices 0,1` or `2,3`); never share chips. (§3)
4. `before_loop` with `--perf-test … -k <case>` pinned; confirm a baseline + buckets. (§4)
5. Local git baseline for the model dir so REVERT/COMMIT work. (§5)
6. `agent.loop --until DECIDE` once to validate the chain; then full `agent.loop`. (§6)
7. Verify wins from `ledger.jsonl` (`delta` < 0 + `result: keep`); never claim a
   speedup the ledger doesn't show. Don't push commits unless explicitly asked.

---

## 8. Safety

- **Two device jobs never share chips** — co-running tt-metal jobs on the same chips
  hang/OOM. Use a free board or wait.
- **COMMIT/REVERT are local git ops** — they never push. Pushing is always a separate,
  explicit step.
- **`.env.agent` holds a live API key** — `chmod 600`, never commit/push it.
- **Don't trust the edit agent's self-report** — trust the measured `delta` in the
  ledger and the PCC gate. A change is "real" only if it's faster AND correct.
