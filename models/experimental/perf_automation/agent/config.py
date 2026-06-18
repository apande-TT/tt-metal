"""Credential loading — `.env.agent` is the ONLY credential source (PLAN section 3.1).

Rules enforced here (not by convention):
  1. Load credentials ONLY from `.env.agent`; never fall back to the shell env.
  2. Fail fast with an actionable message when the file is missing/incomplete.
  3. Map LiteLLM creds to the ANTHROPIC_* vars the SDK process consumes
     (the POC wiring: base url, auth token, api key, small-fast model, and
     telemetry/autoupdater opt-outs).
  4. The key never leaves the process env (never logged/printed/persisted).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

from dotenv import dotenv_values

REQUIRED_KEYS = ("LITELLM_BASE_URL", "LITELLM_API_KEY")

# Exact, actionable prompt (PLAN section 3.1). Carries no secret values.
MISSING_ENV_MESSAGE = (
    "Missing .env.agent — create perf_automation/.env.agent with "
    "LITELLM_BASE_URL=... and LITELLM_API_KEY=... then re-run."
)

# Model roles (PLAN section 3.1). Sub-agents use sonnet; lead model is TBD(model-lead)
# (likely Opus 4.8) — default to sonnet until resolved. Both overridable via .env.agent.
MODEL_ENV_KEYS = {
    "lead": "AGENT_MODEL_LEAD",
    "sub": "AGENT_MODEL_SUB",
    "edit": "AGENT_MODEL_EDIT",
    "structural": "AGENT_MODEL_STRUCTURAL",
}
MODEL_DEFAULTS = {
    "sub": "anthropic/claude-sonnet-4-6",
    "lead": "anthropic/claude-sonnet-4-6",  # TBD(model-lead)
    "edit": "anthropic/claude-haiku-4-5-20251001",  # editing inherits the SUB (haiku) tier; it applies the lead's PLAN spec verbatim
    # structural-edit agent: a COORDINATED multi-op change (shard tensor + program
    # config + consumer) is reasoning-heavy, so it gets the lead (sonnet) tier, not
    # the mechanical haiku editor. Override via AGENT_MODEL_STRUCTURAL.
    "structural": "anthropic/claude-sonnet-4-6",
}


class ConfigError(Exception):
    """Raised when `.env.agent` is absent or incomplete."""


def load_agent_env(env_path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse `.env.agent` (and ONLY that file) and return the resolved config.

    Returns a dict of the file's own keys plus the mapped ANTHROPIC_* vars.
    Never reads the ambient shell environment for the required keys.
    Raises ConfigError (with MISSING_ENV_MESSAGE) when missing/incomplete.
    """
    path = Path(env_path)
    if not path.is_file():
        raise ConfigError(MISSING_ENV_MESSAGE)

    # dotenv_values reads ONLY this file — no shell-env fallback.
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}

    for key in REQUIRED_KEYS:
        if not values.get(key):
            raise ConfigError(MISSING_ENV_MESSAGE)

    api_key = values["LITELLM_API_KEY"]
    resolved: dict[str, str] = dict(values)
    resolved.update(
        {
            "ANTHROPIC_BASE_URL": values["LITELLM_BASE_URL"],
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_API_KEY": api_key,
        }
    )
    return resolved


def get_model(role: str, config: dict[str, str] | None = None) -> str:
    """Resolve the model id for a role ('lead' | 'sub').

    Precedence: AGENT_MODEL_<ROLE> in the resolved `.env.agent` config, else the
    documented default (PLAN section 3.1). Centralized so M3+ call sites never
    invent their own fallback.
    """
    if role not in MODEL_ENV_KEYS:
        raise ValueError(f"unknown model role: {role!r} (expected 'lead' or 'sub')")
    config = config or {}
    override = config.get(MODEL_ENV_KEYS[role])
    if override:
        return override
    if role == "edit":
        # Editing is a MECHANICAL task: the editor applies the lead's localized
        # PLAN spec verbatim, so it inherits the SUB (haiku) model from .env
        # unless AGENT_MODEL_EDIT is set explicitly. The reasoning/localization
        # lives in PLAN (lead) — the editor only transcribes.
        return config.get(MODEL_ENV_KEYS["edit"]) or config.get(MODEL_ENV_KEYS["sub"]) or MODEL_DEFAULTS["edit"]
    return MODEL_DEFAULTS[role]


# Edit-model escalation ladder (cheap-first): APPLY uses rung 0; each REPAIR_CODE attempt
# climbs one rung. haiku -> sonnet -> opus. Spend the cheap model on the easy edits and
# only escalate to the expensive one when the edit keeps failing. Per-rung override via
# .env.agent (AGENT_MODEL_EDIT_1/2/3); the top rung is reused once the ladder is exhausted.
EDIT_LADDER_ENV_KEYS = ("AGENT_MODEL_EDIT_1", "AGENT_MODEL_EDIT_2", "AGENT_MODEL_EDIT_3")
EDIT_LADDER_DEFAULTS = (
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-8",
)


def get_edit_model(attempt: int, config: dict[str, str] | None = None) -> str:
    """Model for the Nth edit attempt on a lever: rung 0 (APPLY) -> haiku, rung 1 (first
    repair) -> sonnet, rung 2+ (later repairs) -> opus. Per-rung override via
    AGENT_MODEL_EDIT_{1,2,3} in .env.agent; capped at the top rung once exhausted."""
    config = config or {}
    rung = max(0, min(int(attempt), len(EDIT_LADDER_DEFAULTS) - 1))
    return config.get(EDIT_LADDER_ENV_KEYS[rung]) or EDIT_LADDER_DEFAULTS[rung]


# Vars injected into the SDK process env. The ANTHROPIC_* creds plus the POC
# wiring: small-fast model (haiku-class internal calls must hit a model the
# proxy serves) and telemetry/autoupdater opt-outs.
SDK_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_SMALL_FAST_MODEL",
)
STATIC_SDK_ENV = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
}


def apply_agent_env(
    env_path: str | os.PathLike[str],
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Load `.env.agent` and inject the SDK env vars into `environ`.

    Injects the ANTHROPIC_* creds, ANTHROPIC_SMALL_FAST_MODEL (= the sub-agent
    model), and the static telemetry/autoupdater opt-outs. Defaults to
    os.environ (the live SDK process env). Returns the resolved config.
    """
    if environ is None:
        environ = os.environ
    resolved = load_agent_env(env_path)
    # Small-fast model = sub-agent model so SDK internal calls hit a served model.
    resolved.setdefault("ANTHROPIC_SMALL_FAST_MODEL", get_model("sub", resolved))
    for key in SDK_ENV_KEYS:
        environ[key] = resolved[key]
    for key, value in STATIC_SDK_ENV.items():
        environ[key] = value
    return resolved
