"""Unit tests for the Homeostasis regulator state."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from interfaces import CommandResult, HomeostasisProtocol
from regulator.state import Homeostasis


def test_homeostasis_protocol_conformance() -> None:
    """Verify Homeostasis instance conforms to HomeostasisProtocol."""
    h = Homeostasis()
    assert isinstance(h, HomeostasisProtocol)
    assert h.dopamine == pytest.approx(0.2)
    assert h.cortisol == pytest.approx(0.0)
    assert h.adenosine == pytest.approx(0.0)
    assert not h.should_lock_slot()
    assert not h.should_terminate_trial()
    assert not h.should_trigger_sleep()


def test_clamping_ranges() -> None:
    """Verify chemical sliders are strictly clamped within [0.0, 1.0]."""
    # Clamping during initialization
    h = Homeostasis(dopamine=2.5, cortisol=-0.5, adenosine=3.0)
    assert h.dopamine == pytest.approx(1.0)
    assert h.cortisol == pytest.approx(0.0)
    assert h.adenosine == pytest.approx(1.0)

    # Clamping on direct assignment
    h.dopamine = -1.0
    assert h.dopamine == pytest.approx(0.0)
    h.dopamine = 1.8
    assert h.dopamine == pytest.approx(1.0)

    h.cortisol = -0.1
    assert h.cortisol == pytest.approx(0.0)
    h.cortisol = 2.0
    assert h.cortisol == pytest.approx(1.0)

    h.adenosine = -0.5
    assert h.adenosine == pytest.approx(0.0)
    h.adenosine = 1.5
    assert h.adenosine == pytest.approx(1.0)


def test_dopamine_spikes_and_lock_threshold() -> None:
    """Verify dopamine spikes on success (exit_code 0) and triggers lock at >= 0.8."""
    h = Homeostasis(dopamine=0.2, dopamine_boost=0.3)
    assert not h.should_lock_slot()

    # First success
    h.on_command_result(0, accumulate_adenosine=False)
    assert h.dopamine == pytest.approx(0.5)
    assert not h.should_lock_slot()

    # Second success: reaches 0.8
    h.on_command_result(0, accumulate_adenosine=False)
    assert h.dopamine == pytest.approx(0.8)
    assert h.should_lock_slot()

    # Third success: clamps at 1.0
    h.on_command_result(0, accumulate_adenosine=False)
    assert h.dopamine == pytest.approx(1.0)
    assert h.should_lock_slot()

    # Verify CommandResult object works seamlessly
    h_cmd = Homeostasis(dopamine=0.2, dopamine_boost=0.6)
    cmd_success = CommandResult(command="make", stdout="done", stderr="", exit_code=0)
    h_cmd.on_command_result(cmd_success, accumulate_adenosine=False)
    assert h_cmd.dopamine == pytest.approx(0.8)
    assert h_cmd.should_lock_slot()


def test_cortisol_repeat_error_termination() -> None:
    """Verify cortisol rises on error, spikes higher on repeat errors, and terminates trial at >= 0.8."""
    h = Homeostasis(
        cortisol=0.0,
        cortisol_boost=0.25,
        cortisol_repeat_boost=0.45,
    )
    assert not h.should_terminate_trial()

    # First error: initial boost
    h.on_command_result(1, is_repeat_error=False, accumulate_adenosine=False)
    assert h.cortisol == pytest.approx(0.25)
    assert not h.should_terminate_trial()

    # Repeat error: repeat boost (0.25 + 0.45 = 0.70)
    h.on_command_result(1, is_repeat_error=True, accumulate_adenosine=False)
    assert h.cortisol == pytest.approx(0.70)
    assert not h.should_terminate_trial()

    # Another repeat error: reaches threshold (0.70 + 0.45 = 1.0 clamped)
    h.on_command_result(1, is_repeat_error=True, accumulate_adenosine=False)
    assert h.cortisol == pytest.approx(1.0)
    assert h.should_terminate_trial()

    # Configurable termination sensitivity test
    h_sensitive = Homeostasis(
        cortisol=0.0,
        cortisol_boost=0.3,
        cortisol_repeat_boost=0.55,
    )
    h_sensitive.on_command_result(1, is_repeat_error=False, accumulate_adenosine=False)
    assert not h_sensitive.should_terminate_trial()
    h_sensitive.on_command_result(1, is_repeat_error=True, accumulate_adenosine=False)
    assert h_sensitive.cortisol == pytest.approx(0.85)
    assert h_sensitive.should_terminate_trial()


def test_adenosine_accumulation_and_sleep_trigger() -> None:
    """Verify adenosine accumulates per tool call and triggers sleep cycle at >= 1.0."""
    h = Homeostasis(adenosine=0.0, adenosine_step=0.1)
    assert not h.should_trigger_sleep()

    # 9 tool calls: reaches 0.9
    for i in range(9):
        val = h.record_tool_call()
        assert val == pytest.approx((i + 1) * 0.1)
        assert not h.should_trigger_sleep()

    # 10th tool call: reaches 1.0 and triggers sleep
    h.record_tool_call()
    assert h.adenosine == pytest.approx(1.0)
    assert h.should_trigger_sleep()

    # Extra tool call: clamped at 1.0
    h.record_tool_call()
    assert h.adenosine == pytest.approx(1.0)

    # Reset adenosine clears sleep trigger
    h.reset_adenosine()
    assert h.adenosine == pytest.approx(0.0)
    assert not h.should_trigger_sleep()


def test_command_result_accumulates_adenosine_by_default() -> None:
    """Verify on_command_result accumulates adenosine by default."""
    h = Homeostasis(adenosine=0.0, adenosine_step=0.2)
    h.on_command_result(0)
    assert h.adenosine == pytest.approx(0.2)
    h.on_command_result(1)
    assert h.adenosine == pytest.approx(0.4)


def test_decay_math() -> None:
    """Verify step_decay decays elevated levels toward their baselines."""
    h = Homeostasis(
        dopamine=0.8,
        cortisol=0.6,
        dopamine_baseline=0.2,
        cortisol_baseline=0.0,
        dopamine_decay=0.1,
        cortisol_decay=0.1,
    )

    h.step_decay()
    assert h.dopamine == pytest.approx(0.7)
    assert h.cortisol == pytest.approx(0.5)

    # Repeated decay
    for _ in range(10):
        h.step_decay()

    # Should plateau at respective baselines, never going below
    assert h.dopamine == pytest.approx(0.2)
    assert h.cortisol == pytest.approx(0.0)

    # If dopamine was suppressed below baseline, it recovers toward baseline
    h.dopamine = 0.05
    h.step_decay()
    assert h.dopamine == pytest.approx(0.15)
    h.step_decay()
    assert h.dopamine == pytest.approx(0.2)


def test_sleep_relaxes_cortisol() -> None:
    """Verify reset_adenosine optionally relieves stress/cortisol."""
    h = Homeostasis(cortisol=0.8, adenosine=1.0)
    h.reset_adenosine(relax_cortisol=True)
    assert h.adenosine == pytest.approx(0.0)
    assert h.cortisol == pytest.approx(0.4)

    # When relax_cortisol is False
    h.cortisol = 0.8
    h.adenosine = 1.0
    h.reset_adenosine(relax_cortisol=False)
    assert h.adenosine == pytest.approx(0.0)
    assert h.cortisol == pytest.approx(0.8)


def test_snapshot_representation() -> None:
    """Verify snapshot dictionary format and accuracy."""
    h = Homeostasis(dopamine=0.85, cortisol=0.2, adenosine=1.0)
    snap = h.snapshot()
    assert snap == {
        "dopamine": pytest.approx(0.85),
        "cortisol": pytest.approx(0.2),
        "adenosine": pytest.approx(1.0),
        "should_lock_slot": True,
        "should_terminate_trial": False,
        "should_trigger_sleep": True,
    }
