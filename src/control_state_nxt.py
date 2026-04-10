"""
Control-state machine and PID profile loader for run_agent_nxt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class ControlState(str, Enum):
    """High-level driving state for PID profile selection."""

    GO = "GO"
    STOP = "STOP"


@dataclass(frozen=True)
class ControlTransitionConfig:
    """Configuration for CoT-based control-state transitions."""

    confirm_count: int
    stop_patterns: tuple[re.Pattern[str], ...]
    go_patterns: tuple[re.Pattern[str], ...]


class ControlStateMachine:
    """State machine driven by CoT text classification."""

    def __init__(self, transition_config: ControlTransitionConfig):
        self._cfg = transition_config
        self.state = ControlState.GO
        self._go_count = 0
        self._stop_count = 0

    @property
    def confirm_count(self) -> int:
        return self._cfg.confirm_count

    def update_from_text(self, text: str) -> bool:
        """
        Update the state machine from one CoT sample.

        Returns:
            True when the state changes; otherwise False.
        """
        direction = self._classify(text)
        if direction == "STOP":
            self._stop_count += 1
            self._go_count = 0
        elif direction == "GO":
            self._go_count += 1
            self._stop_count = 0
        else:
            self._go_count = 0
            self._stop_count = 0

        prev_state = self.state
        if self.state == ControlState.GO and self._stop_count >= self._cfg.confirm_count:
            self.state = ControlState.STOP
            self._stop_count = 0
        elif self.state == ControlState.STOP and self._go_count >= self._cfg.confirm_count:
            self.state = ControlState.GO
            self._go_count = 0
        return self.state != prev_state

    def _classify(self, text: str) -> Optional[str]:
        lowered = (text or "").strip().lower()
        if not lowered:
            return None
        if _matches_any(lowered, self._cfg.stop_patterns):
            return "STOP"
        if _matches_any(lowered, self._cfg.go_patterns):
            return "GO"
        return None


def _matches_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.search(text) is not None for p in patterns)


def load_pid_state_profile_bundle(path: str) -> Dict[str, Any]:
    """Load and minimally validate the state-profile JSON file."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as f:
        bundle = json.load(f)
    if not isinstance(bundle, dict):
        raise ValueError("PID state profile JSON must be an object.")
    if "states" not in bundle or not isinstance(bundle["states"], dict):
        raise ValueError("PID state profile JSON must include 'states' object.")
    return bundle


def build_transition_config(bundle: Dict[str, Any]) -> ControlTransitionConfig:
    transition = bundle.get("transition", {})
    if not isinstance(transition, dict):
        transition = {}
    confirm_count = int(transition.get("confirm_count", 2))
    if confirm_count <= 0:
        raise ValueError("transition.confirm_count must be >= 1")

    stop_tokens = transition.get(
        "stop_patterns",
        [
            r"\bstop\b",
            r"\bbrake\b",
            r"\bdecelerat(e|ing)\b",
            r"\bslow\s+down\b",
            r"\bslow\s+to\b",
            r"\bprepare\s+to\s+stop\b",
            r"\bwait\b",
            r"\bhold\b",
            r"\byield\b",
            r"\bred light\b",
            r"\byellow traffic light\b",
            r"\btraffic signal\b",
        ],
    )
    go_tokens = transition.get(
        "go_patterns",
        [
            r"\bkeep\b",
            r"\bgo\b",
            r"\bproceed\b",
            r"\bcontinue\b",
            r"\bmove\b",
            r"\bdrive\b",
            r"\baccelerat(e|ing)\b",
            r"\bresume\b",
        ],
    )
    if not isinstance(stop_tokens, list) or not isinstance(go_tokens, list):
        raise ValueError("transition stop/go patterns must be arrays")

    return ControlTransitionConfig(
        confirm_count=confirm_count,
        stop_patterns=tuple(re.compile(str(p), re.IGNORECASE) for p in stop_tokens),
        go_patterns=tuple(re.compile(str(p), re.IGNORECASE) for p in go_tokens),
    )


def resolve_state_overrides(
    bundle: Dict[str, Any],
    state: ControlState,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    """
    Resolve PID overrides for the given state.

    Missing keys are filled from fallback.
    """
    states = bundle.get("states", {})
    if not isinstance(states, dict):
        states = {}

    merged: Dict[str, float] = dict(fallback)
    state_obj = states.get(state.value, {})
    if isinstance(state_obj, dict):
        for key, value in state_obj.items():
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
    return merged
