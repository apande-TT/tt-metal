# Task Prompt — feature/tt-hw-planner

> Paste the task description below, then send the whole block to the agent.
> The constraints and the push step are non-negotiable and apply to every task on this branch.

---

## Task

<!-- Replace this line with the issue/feature/fix description.
     Include links, error text, or reproduction steps if you have them. -->

## Constraints (non-negotiable)

1. **Don't break the code.**
   Existing tests, builds, and call sites must keep passing. If a change forces a deprecation or behavior change, call it out explicitly — never break things silently. Prefer additive changes over edits to shared signatures.

2. **Don't add duplication.**
   Before writing new logic, check whether a helper/pattern already exists. Reuse it, or extract a shared helper if needed. No copy-pasted blocks, no near-identical functions, no parallel implementations of the same idea.

3. **Don't add hardcoded names.**
   No literal device names, hostnames, IPs, user handles, branch names, model IDs, file paths, or magic strings embedded in code. Use config, constants, lookups, or parameters instead. If a value is environment- or instance-specific, it belongs in config — not in source.

   **This includes model/stack stage names.** Do not hardcode names of model components or inference stages — e.g. `decode`, `prefill`, `encoder`, `decoder`, `language_model`, `audio_tower`, layer names, op names, or any similar identifiers — as string literals, dict keys, or `if stage == "..."` branches. These must be **discovered from the model itself** (walk the module graph, read the checkpoint's metadata, or query the model's own schema/registry) and referenced by what the model reports, not by a string you typed. If the model renames a stage tomorrow, the code should still work because it never assumed the name.

4. **Check the flow — don't introduce new bugs while solving this one.**
   Before declaring done, trace every affected code path end-to-end:
   - All callers of changed functions still behave correctly.
   - Edge cases and error paths still work (or are explicitly improved).
   - No regressions, no dead branches left behind, no silent behavior changes.
   - Re-run the relevant tests/lints and confirm green.

## When done

1. Run the relevant tests / checks and confirm they pass.
2. Commit with a clear message describing **what** changed and **why**.
3. **Push to `feature/tt-hw-planner`** — this is your tool branch. Always push here when the work is complete. Do not leave commits stranded locally, and do not push to any other branch unless explicitly told to. **Commit and push as `apande-TT` only** (`user.name=apande-TT`, GitHub credentials for that login). Do not use any other git name or GitHub account — including a full-name identity, a bot, or a cloud-agent account. If the current `user.name` is not `apande-TT`, set it before the commit/push.
