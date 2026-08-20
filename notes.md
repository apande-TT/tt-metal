# Repo Assist Memory

## Run 2026-08-18 (run 32173957712)
- Attempted normal scheduled workflow (no /repo-assist command instructions given).
- Blocker: GitHub MCP read tools (list_issues, pull_request_read, search_issues, get) returned
  "filtered by secrecy policy — agent not authorized to access private-scoped data" for ALL
  issues and PRs in apande-TT/tt-metal, even though items are confirmed to exist (issue #1,
  PR #24 referenced by tools but blocked). search_issues/search_pull_requests also hit GitHub
  search API rate limits (30/min) independently of the secrecy filtering.
- Could not perform Task 1 (triage/label), Task 2 (investigate/comment), Task 3 (fix via PR),
  Task 5 (maintain repo-assist PRs), Task 6 (stale PR nudges), Task 7 (welcome new contributors),
  or Task 8 (monthly summary issue) because none of these are possible without reading issue/PR
  state, and the monthly summary issue itself could not be searched/read/created-checked.
- No repo-memory history existed prior to this run (first run).
- Action taken: called `report_incomplete` / `missing_data` to flag the environment restriction
  rather than guessing at repo state or fabricating actions.
- Next run: re-check whether the secrecy-policy filtering on issues/PRs has been lifted before
  resuming normal round-robin tasks. If still blocked, re-report rather than retry repeatedly.

## Run 2026-08-19 (run 32206638332)
- Re-checked whether secrecy-policy filtering on issues/PRs was lifted, per prior run's note.
- Still blocked: `list_issues` (owner apande-TT, repo tt-metal, state OPEN) returned `[]` visible
  items, with all 5 sampled issues (#33-#37) explicitly filtered: "Resource has secrecy
  requirements that agent doesn't meet. The agent is not authorized to access private-scoped data."
- This confirms the blocker is persistent/environmental, not transient. No issue/PR reads are
  possible, so Tasks 1,2,3,5,6,7,8 (monthly summary requires searching/reading the summary issue)
  remain impossible this run too.
- Action: called report_incomplete again (2nd consecutive run blocked). Did not retry repeatedly
  beyond one verification call, per guidance not to hammer a known-broken path.
- Next run: check again before resuming; if blocked a 3rd time, consider this a standing
  environment limitation worth escalating outside the automated loop rather than re-checking
  every run indefinitely.

## Run 2026-08-19 (run 32226372452)
- 3rd consecutive run checking the secrecy-policy blocker on issue/PR reads.
- Still blocked: list_issues (state OPEN, apande-TT/tt-metal) returned 0 visible items, same
  5 issues (#33-#37) explicitly filtered with "secrecy requirements ... not authorized to
  access private-scoped data."
- Confirms this is a standing environment/config limitation (not transient), now observed
  across 3 separate runs on 2 different days. All tasks (1,2,3,5,6,7,8) remain impossible
  since every one requires reading issue/PR state via the github MCP tool.
- Action: called report_incomplete once (no repeated hammering). Escalating verbally in the
  report that this needs a maintainer/infra fix (secrecy scope on the MCP server / token),
  not another automated retry.
- Next run: do NOT re-probe list_issues again on the very next run by default — instead check
  first for any maintainer comment/instruction on the monthly summary issue if it becomes
  readable, or any other signal that the secrecy policy config changed, before assuming this
  is still broken. If still unreadable, just call report_incomplete/noop without re-listing.

## Run 2026-08-19 (run 32256696045)
- 4th consecutive run checking the secrecy-policy blocker on issue/PR reads.
- Still blocked: list_issues (state OPEN, apande-TT/tt-metal) returned 0 visible items; issues
  #34-#38 all explicitly filtered with "secrecy requirements ... not authorized to access
  private-scoped data." Confirms standing environment/config limitation across 4 runs, 2+ days.
- All tasks (1,2,3,5,6,7,8) remain impossible — every one requires reading issue/PR state via
  the github MCP tool, which is fully blocked.
- Action: called report_incomplete once; no repeated hammering of list_issues/search tools.
- Next run: do NOT re-probe repeatedly by default. Check once for signal the secrecy policy
  changed; if still blocked, just report_incomplete/noop without extra tool calls.

## Run 2026-08-19 (run 32290064024)
- 5th consecutive run checking the secrecy-policy blocker on issue/PR reads.
- Still blocked: github-issue_read get on issue #1 (apande-TT/tt-metal) returned
  "filtered by secrecy policy ... not authorized to access private-scoped data."
- Confirms standing environment/config limitation across 5 runs, 2+ days, unchanged.
- All tasks (1,2,3,5,6,7,8) remain impossible — every one requires reading issue/PR state via
  the github MCP tool, which is fully blocked at the secrecy-policy layer (not rate limits).
- Action: called report_incomplete once; did not repeatedly hammer read tools.
- Next run: do NOT re-probe every single run. Only re-check if there is external signal this
  changed (e.g. a maintainer note elsewhere). Otherwise treat as a known standing blocker and
  report_incomplete/noop quickly without extensive re-verification.

## Run 2026-08-20 (run 32322640012)
- 6th consecutive run checking the secrecy-policy blocker on issue/PR reads.
- Still blocked: github-issue_read get on issue #1 (apande-TT/tt-metal) returned identical
  "filtered by secrecy policy ... not authorized to access private-scoped data" error.
- Confirms standing environment/config limitation across 6 runs, 2+ days, unchanged. No
  instructions were given via command mode (instructions empty), so normal scheduled workflow
  applied but remains impossible — all tasks (1,2,3,5,6,7,8) require reading issue/PR state.
- Action: single verification read call only (no hammering), then report_incomplete.
- Next run: continue light-touch check (1 read call) before reporting; escalate to a human
  if this persists much longer, as it's now a multi-day standing blocker needing infra/maintainer
  attention (MCP server secrecy scope/token config), not something repo-assist can self-resolve.

## Run 2026-08-20 (run 32342615965)
- 7th consecutive run checking the secrecy-policy blocker on issue/PR reads.
- Still blocked: github-issue_read get on issue #1 (apande-TT/tt-metal) returned identical
  "filtered by secrecy policy ... not authorized to access private-scoped data" error.
- Confirms standing environment/config limitation across 7 runs, 2+ days, unchanged. No
  command-mode instructions given (empty), so normal scheduled workflow applied but remains
  impossible — all tasks (1,2,3,5,6,7,8) require reading issue/PR state via github MCP tool.
- Action: single verification read call only (no hammering), then report_incomplete.
- Next run: continue light-touch check (1 read call) before reporting. This is now a week-long
  standing blocker needing infra/maintainer attention (MCP server secrecy scope/token config).
