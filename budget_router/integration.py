"""Optional 0xClaw entry point around the unchanged BudgetRouter runtime."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .budget_router import BudgetRouterRuntime, apply_budget_decision, log_budget_decision


class BudgetRouterPlugin:
    def __init__(self, *, root: Path, workspace: Path):
        model_path = os.environ.get("OXCLAW_BUDGET_ROUTER_MODEL_PATH", "")
        adapter_path = os.environ.get("OXCLAW_BUDGET_ROUTER_ADAPTER_PATH", "")
        if not model_path or not adapter_path:
            raise ValueError(
                "Set OXCLAW_BUDGET_ROUTER_MODEL_PATH and OXCLAW_BUDGET_ROUTER_ADAPTER_PATH "
                "to the downloaded base model and LoRA adapter directories."
            )
        mode = os.environ.get("OXCLAW_BUDGET_ROUTER_MODE", "enforce")
        if mode not in {"enforce", "shadow"}:
            raise ValueError("OXCLAW_BUDGET_ROUTER_MODE must be enforce or shadow")
        self.runtime = BudgetRouterRuntime(
            SimpleNamespace(enabled=True, mode=mode, model_path=model_path,
                            adapter_path=adapter_path, max_new_tokens=48, schema_aware=True),
            root=root, workspace=workspace,
        )
        self.log_path = workspace / "hackathon" / "budget_router.jsonl"

    def prepare(self, phase: str, command: str, profile: Any, context: dict) -> Any:
        if profile is None:
            return profile
        decision = self.runtime.decide(phase, command)
        expected = "documentation" if phase == "doc" else phase
        if decision.is_model_decision and decision.phase != expected:
            raise RuntimeError(f"BudgetRouter returned {decision.phase!r} for phase {expected!r}")
        if not decision.is_model_decision:
            warnings.warn(f"BudgetRouter skipped: {decision.error or decision.source}", stacklevel=2)
        effective = apply_budget_decision(profile, decision, mode=self.runtime.mode)
        log_budget_decision(
            self.log_path, phase=phase, command=command, decision=decision,
            base_profile=profile, effective_profile=effective, mode=self.runtime.mode,
        )
        context["budget_decision"] = decision
        context["budget_mode"] = self.runtime.mode
        return effective

    def frame(self, phase: str, message: str, context: dict) -> str:
        return message
