"""Homeostatic regulator state tracking neurotransmitter / chemical sliders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from interfaces import HomeostasisProtocol


@dataclass
class Homeostasis(HomeostasisProtocol):
    """Homeostatic state controller managing chemical sliders in [0.0, 1.0].

    Sliders:
    - dopamine: spikes on exit code 0; locks memory slot when >= 0.8
    - cortisol: rises on errors, spikes higher on repeat errors; terminates trial loop at >= 0.8
    - adenosine: accumulates per tool call; triggers sleep cycle at >= 1.0
    """

    dopamine: float = 0.2
    cortisol: float = 0.0
    adenosine: float = 0.0

    # Boost and decay parameters
    dopamine_boost: float = 0.3
    dopamine_decay: float = 0.05
    cortisol_boost: float = 0.25
    cortisol_repeat_boost: float = 0.45
    cortisol_decay: float = 0.05
    adenosine_step: float = 0.1

    # Baselines
    dopamine_baseline: float = 0.2
    cortisol_baseline: float = 0.0

    # Thresholds
    lock_threshold: float = 0.8
    terminate_threshold: float = 0.8
    sleep_threshold: float = 1.0

    def __setattr__(self, name: str, value: Any) -> None:
        """Enforce strict [0.0, 1.0] clamping for chemical sliders."""
        if name in ("dopamine", "cortisol", "adenosine") and isinstance(value, (int, float)):
            value = round(max(0.0, min(1.0, float(value))), 6)
        super().__setattr__(name, value)

    def record_tool_call(self, step: Optional[float] = None) -> float:
        """Accumulate adenosine per tool call. Returns updated adenosine level."""
        delta = self.adenosine_step if step is None else step
        self.adenosine += delta
        return self.adenosine

    def on_command_result(
        self,
        exit_code: int | Any,
        is_repeat_error: bool = False,
        accumulate_adenosine: bool = True,
    ) -> None:
        """Update chemical levels based on command execution result."""
        # Normalize exit_code if a CommandResult object was passed
        code = getattr(exit_code, "binary_exit_code", getattr(exit_code, "exit_code", exit_code))
        try:
            numeric_code = int(code)
        except (TypeError, ValueError):
            numeric_code = 1

        if numeric_code == 0:
            # Command succeeded: dopamine spikes, cortisol relaxes toward baseline
            self.dopamine += self.dopamine_boost
            if self.cortisol > self.cortisol_baseline:
                self.cortisol = max(self.cortisol_baseline, self.cortisol - self.cortisol_decay)
        else:
            # Command failed: cortisol rises (higher on repeat errors), dopamine decays
            if is_repeat_error:
                self.cortisol += self.cortisol_repeat_boost
            else:
                self.cortisol += self.cortisol_boost

            if self.dopamine > self.dopamine_baseline:
                self.dopamine = max(self.dopamine_baseline, self.dopamine - self.dopamine_decay)

        if accumulate_adenosine:
            self.record_tool_call()

    def step_decay(self) -> None:
        """Apply homeostatic decay toward baseline for dopamine and cortisol."""
        if self.dopamine > self.dopamine_baseline:
            self.dopamine = max(self.dopamine_baseline, self.dopamine - self.dopamine_decay)
        elif self.dopamine < self.dopamine_baseline:
            self.dopamine = min(self.dopamine_baseline, self.dopamine + self.dopamine_decay)

        if self.cortisol > self.cortisol_baseline:
            self.cortisol = max(self.cortisol_baseline, self.cortisol - self.cortisol_decay)
        elif self.cortisol < self.cortisol_baseline:
            self.cortisol = min(self.cortisol_baseline, self.cortisol + self.cortisol_decay)

    def should_lock_slot(self) -> bool:
        """Check if dopamine has reached or exceeded the slot lock threshold (>= 0.8)."""
        return self.dopamine >= self.lock_threshold

    def should_terminate_trial(self) -> bool:
        """Check if cortisol has reached or exceeded trial termination threshold (>= 0.8)."""
        return self.cortisol >= self.terminate_threshold

    def should_trigger_sleep(self) -> bool:
        """Check if adenosine has reached or exceeded sleep threshold (>= 1.0)."""
        return self.adenosine >= self.sleep_threshold

    def reset_adenosine(self, relax_cortisol: bool = True) -> None:
        """Reset adenosine to 0.0 and optionally relax cortisol after sleep."""
        self.adenosine = 0.0
        if relax_cortisol:
            self.cortisol = max(self.cortisol_baseline, self.cortisol * 0.5)

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot dictionary containing current values and threshold flags."""
        return {
            "dopamine": self.dopamine,
            "cortisol": self.cortisol,
            "adenosine": self.adenosine,
            "should_lock_slot": self.should_lock_slot(),
            "should_terminate_trial": self.should_terminate_trial(),
            "should_trigger_sleep": self.should_trigger_sleep(),
        }
