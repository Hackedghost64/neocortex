"""Comprehensive End-to-End Integration Tests for Project Genesis.

Verifies end-to-end integration across all four core modules:
1. Body (LocalSandboxRunner / DockerSandboxRunner)
2. Regulator (Homeostasis chemical sliders)
3. Memory Network (LoRAManager LoRA slots in SQLite)
4. Sleep & Consolidation (git diff extraction, TrainingPair synthesis, 3-epoch fine-tuning)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Sequence
import pytest

from engine.lora_manager import LoRAManager
from interfaces import (
    CommandResult,
    HomeostasisProtocol,
    LoRAManagerProtocol,
    SandboxRunnerProtocol,
    SlotRecord,
    TrialOutcome,
)
from orchestrator import (
    CALC_SOURCE,
    GenesisOrchestrator,
    LocalSandboxRunner,
    TASK_DESCRIPTION,
    TEST_CALC_SOURCE,
    default_policy_fn,
    provision_trial,
    run_trial,
)
from regulator.state import Homeostasis
from sandbox.docker_runner import DockerSandboxRunner


# ---------------------------------------------------------------------------
# Fixtures & Helper Utilities
# ---------------------------------------------------------------------------


def is_docker_with_git_available() -> bool:
    """Check if the Docker daemon is responsive AND has an image with git installed."""
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "python:3.11-slim", "which", "git"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


@pytest.fixture
def local_sandbox():
    """Create and automatically clean up a LocalSandboxRunner instance."""
    runner = LocalSandboxRunner(auto_cleanup=True)
    runner.start()
    try:
        yield runner
    finally:
        runner.stop()


@pytest.fixture
def in_memory_lora_manager():
    """Create a LoRAManager backed by an in-memory SQLite database."""
    manager = LoRAManager(db_path=":memory:", use_mock=True)
    # Register an arithmetic slot for routing
    manager.init_slot(
        slot_id="slot_math",
        centroid_vector=[1.0] + [0.0] * 15,
        salience=0.5,
    )
    # Register an unrelated slot
    manager.init_slot(
        slot_id="slot_io",
        centroid_vector=[0.0] * 8 + [1.0] + [0.0] * 7,
        salience=0.5,
    )
    return manager


# ---------------------------------------------------------------------------
# 1. Protocol Conformance & Sandbox Basics
# ---------------------------------------------------------------------------


def test_sandbox_protocol_conformance(local_sandbox: LocalSandboxRunner) -> None:
    """Verify LocalSandboxRunner conforms to SandboxRunnerProtocol."""
    assert isinstance(local_sandbox, SandboxRunnerProtocol)
    assert local_sandbox.is_running is True
    assert os.path.isdir(local_sandbox.workspace)


def test_local_sandbox_command_execution(local_sandbox: LocalSandboxRunner) -> None:
    """Verify command execution and stdout/stderr capture in local runner."""
    res = local_sandbox.run_command("echo 'hello genesis'")
    assert res.exit_code == 0
    assert "hello genesis" in res.stdout
    assert res.succeeded is True
    assert res.binary_exit_code == 0

    fail_res = local_sandbox.run_command("nonexistent_command_xyz_123 2>&1")
    assert fail_res.exit_code != 0
    assert fail_res.succeeded is False
    assert fail_res.binary_exit_code == 1


# ---------------------------------------------------------------------------
# 2. Trial Provisioning Tests
# ---------------------------------------------------------------------------


def test_provision_trial_local_git_repo(local_sandbox: LocalSandboxRunner) -> None:
    """Verify trial provisioning injects broken files and initializes git baseline commit."""
    files = provision_trial(local_sandbox, workdir="/home/trials")
    assert "task.md" in files
    assert "calc.py" in files
    assert "test_calc.py" in files

    # Verify files physically exist on disk
    calc_path = os.path.join(local_sandbox.workspace, "calc.py")
    test_calc_path = os.path.join(local_sandbox.workspace, "test_calc.py")
    task_path = os.path.join(local_sandbox.workspace, "task.md")

    assert os.path.isfile(calc_path)
    assert os.path.isfile(test_calc_path)
    assert os.path.isfile(task_path)

    with open(calc_path, "r", encoding="utf-8") as f:
        assert f.read() == CALC_SOURCE

    # Verify git repository was initialized and baseline committed
    git_status = local_sandbox.run_command("git status --porcelain")
    assert git_status.exit_code == 0
    assert git_status.stdout.strip() == ""  # Working tree is clean after baseline commit

    git_log = local_sandbox.run_command("git log -1 --pretty=%B")
    assert git_log.exit_code == 0
    assert "Baseline broken scenario" in git_log.stdout

    # Verify initial git diff against HEAD is empty
    diff_res = local_sandbox.run_command("git diff HEAD")
    assert diff_res.exit_code == 0
    assert diff_res.stdout.strip() == ""


def test_baseline_broken_calc_fails(local_sandbox: LocalSandboxRunner) -> None:
    """Verify requirement 1: baseline calc.py causes test_calc.py to exit with non-zero code."""
    provision_trial(local_sandbox, workdir="/home/trials")

    # Run the broken test directly
    res = local_sandbox.run_command("python3 test_calc.py")
    assert res.exit_code != 0
    assert "AssertionError" in res.stderr or "AssertionError" in res.stdout


# ---------------------------------------------------------------------------
# 3. Full End-to-End Consolidation, Sleep Cycle & Slot Locking
# ---------------------------------------------------------------------------


def test_wake_cycle_full_e2e_consolidation(
    local_sandbox: LocalSandboxRunner,
    in_memory_lora_manager: LoRAManager,
) -> None:
    """Verify requirements 1, 2, and 3:

    1. Broken calc.py fails step 0 (raising cortisol & adenosine).
    2. Applying patch makes test_calc.py exit with code 0 on step 1.
    3. Dopamine boosts by +0.4, locks slot when dopamine >= 0.8,
       runs 3-epoch sleep cycle, updates centroid in SQLite, extracts
       clean git diff, and resets adenosine to 0.0.
    """
    # Start with dopamine=0.5 so +0.4 boost reaches 0.9 (>= 0.8 lock threshold)
    regulator = Homeostasis(dopamine=0.5, cortisol=0.0, adenosine=0.0)

    query_vec = [0.95] + [0.05] * 15  # Matches slot_math best

    orchestrator = GenesisOrchestrator(
        sandbox=local_sandbox,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir=local_sandbox.workspace,
    )

    # Multi-step policy: step 0 runs test (fails), step 1 fixes calc.py and tests (passes)
    def test_policy(step: int, task: str, last_result: Optional[CommandResult]) -> str:
        if step == 0:
            return "python3 test_calc.py"
        return (
            "python3 -c \"with open('calc.py', 'w') as f: f.write('def add(a, b):\\n    return a + b\\n')\" && "
            "python3 test_calc.py"
        )

    outcome = orchestrator.run_trial(
        query_embedding=query_vec,
        max_steps=5,
        policy_fn=test_policy,
    )

    # 1. Verification of trial outcome metadata
    assert outcome.status == "consolidated"
    assert outcome.steps_taken == 2
    assert outcome.final_exit_code == 0
    assert outcome.routed_slot_id == "slot_math"
    assert outcome.similarity_score > 0.8

    # 2. Verification of chemical slider updates
    # Dopamine: 0.5 initially -> decayed slightly or boosted: 0.5 - 0.05 + 0.40 = 0.85 (or 0.90)
    assert outcome.dopamine >= 0.80
    # Adenosine: accumulated on tool calls then reset to 0.0 by sleep cycle
    assert outcome.adenosine == pytest.approx(0.0)
    # Cortisol: rose on step 0 error, relaxed after sleep
    assert outcome.cortisol < 0.8

    # 3. Verification of slot locking
    assert outcome.slot_locked is True
    slot_record = in_memory_lora_manager.get_slot("slot_math")
    assert slot_record is not None
    assert slot_record.is_locked is True

    # 4. Verification of sleep report & clean git diff
    assert outcome.sleep_report is not None
    assert outcome.sleep_report["status"] == "success"
    diff_text = outcome.sleep_report["diff"]
    assert "-    return a + b + 1" in diff_text
    assert "+    return a + b" in diff_text

    # 5. Verification of 3-epoch fine-tuning pass
    tune_result = outcome.sleep_report["tune_result"]
    assert tune_result["status"] == "completed"
    assert tune_result["epochs"] == 3
    assert len(tune_result["losses"]) == 3

    # 6. Verification of centroid update in SQLite
    assert slot_record.centroid_vector != [1.0] + [0.0] * 15

    # 7. Verification of dictionary access / helper properties
    assert outcome["status"] == "consolidated"
    assert outcome.get("routed_slot_id") == "slot_math"
    outcome_dict = outcome.to_dict()
    assert isinstance(outcome_dict, dict)
    assert outcome_dict["final_exit_code"] == 0


def test_slot_not_locked_when_dopamine_under_threshold(
    local_sandbox: LocalSandboxRunner,
    in_memory_lora_manager: LoRAManager,
) -> None:
    """Verify that when dopamine < 0.8, consolidation succeeds but slot remains unlocked."""
    # Start with dopamine=0.2 baseline. +0.4 boost makes it 0.60 (< 0.8 lock threshold).
    regulator = Homeostasis(dopamine=0.2, cortisol=0.0, adenosine=0.0)

    orchestrator = GenesisOrchestrator(
        sandbox=local_sandbox,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir=local_sandbox.workspace,
    )

    # 1-step immediate fix policy
    def single_step_pass_policy(step: int, task: str, last_result: Optional[CommandResult]) -> str:
        return (
            "python3 -c \"with open('calc.py', 'w') as f: f.write('def add(a, b):\\n    return a + b\\n')\" && "
            "python3 test_calc.py"
        )

    outcome = orchestrator.run_trial(
        query_embedding=[1.0] + [0.0] * 15,
        max_steps=3,
        policy_fn=single_step_pass_policy,
    )

    assert outcome.status == "consolidated"
    assert outcome.final_exit_code == 0
    assert outcome.dopamine == pytest.approx(0.6)  # 0.2 + 0.4 = 0.60
    assert outcome.slot_locked is False

    slot_record = in_memory_lora_manager.get_slot("slot_math")
    assert slot_record is not None
    assert slot_record.is_locked is False


# ---------------------------------------------------------------------------
# 4. Repeat Errors Spiking Cortisol & Early Termination
# ---------------------------------------------------------------------------


def test_repeat_errors_spike_cortisol_and_terminate(
    local_sandbox: LocalSandboxRunner,
    in_memory_lora_manager: LoRAManager,
) -> None:
    """Verify requirement 4: repeat errors spike cortisol >= 0.8 and terminate trial early."""
    regulator = Homeostasis(
        dopamine=0.2,
        cortisol=0.0,
        adenosine=0.0,
        cortisol_boost=0.25,
        cortisol_repeat_boost=0.45,
    )

    orchestrator = GenesisOrchestrator(
        sandbox=local_sandbox,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir=local_sandbox.workspace,
    )

    # Policy that repeatedly executes the broken test without fixing it
    def failing_policy(step: int, task: str, last_result: Optional[CommandResult]) -> str:
        return "python3 test_calc.py"

    outcome = orchestrator.run_trial(
        query_embedding=[1.0] + [0.0] * 15,
        max_steps=10,
        policy_fn=failing_policy,
    )

    assert outcome.status == "terminated_cortisol"
    # Step 0: boost 0.25 -> 0.25
    # Step 1: repeat boost 0.45 -> 0.70
    # Step 2: repeat boost 0.45 -> 1.0 (>= 0.8 terminate threshold)
    assert outcome.steps_taken == 3
    assert outcome.final_exit_code != 0
    assert outcome.cortisol >= 0.80
    assert outcome.sleep_report is None
    assert outcome.slot_locked is False


def test_adenosine_accumulation_triggers_sleep_on_prolonged_trial(
    local_sandbox: LocalSandboxRunner,
    in_memory_lora_manager: LoRAManager,
) -> None:
    """Verify that if adenosine reaches >= 1.0 before cortisol terminates, sleep is triggered."""
    # Start with high adenosine to reach 1.0 on tool calls
    regulator = Homeostasis(
        dopamine=0.2,
        cortisol=0.0,
        adenosine=0.85,
        adenosine_step=0.1,
        cortisol_boost=0.1,  # Low cortisol boost so it doesn't terminate before adenosine
        cortisol_repeat_boost=0.1,
    )

    orchestrator = GenesisOrchestrator(
        sandbox=local_sandbox,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir=local_sandbox.workspace,
    )

    outcome = orchestrator.run_trial(
        query_embedding=[1.0] + [0.0] * 15,
        max_steps=5,
        policy_fn=lambda step, task, res: "python3 test_calc.py",
    )

    assert outcome.status == "sleep_triggered_adenosine"
    assert outcome.adenosine >= 1.0


def test_max_steps_exceeded(
    local_sandbox: LocalSandboxRunner,
    in_memory_lora_manager: LoRAManager,
) -> None:
    """Verify that when max_steps is exhausted without termination, status is max_steps_exceeded."""
    regulator = Homeostasis(
        dopamine=0.2,
        cortisol=0.0,
        adenosine=0.0,
        cortisol_boost=0.01,
        cortisol_repeat_boost=0.01,
        adenosine_step=0.01,
    )

    orchestrator = GenesisOrchestrator(
        sandbox=local_sandbox,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir=local_sandbox.workspace,
    )

    outcome = orchestrator.run_trial(
        query_embedding=[1.0] + [0.0] * 15,
        max_steps=2,
        policy_fn=lambda step, task, res: "python3 test_calc.py",
    )

    assert outcome.status == "max_steps_exceeded"
    assert outcome.steps_taken == 2


# ---------------------------------------------------------------------------
# 5. Default Policy and Convenience Function Verification
# ---------------------------------------------------------------------------


def test_default_policy_fn_behavior() -> None:
    """Verify default_policy_fn step 0 runs test, and step 1 writes patch."""
    cmd_0 = default_policy_fn(0, "task")
    assert cmd_0 == "python3 test_calc.py"

    cmd_1 = default_policy_fn(1, "task")
    assert "calc.py" in cmd_1
    assert "return a + b" in cmd_1
    assert "python3 test_calc.py" in cmd_1


def test_run_trial_convenience_function(local_sandbox: LocalSandboxRunner) -> None:
    """Verify top-level run_trial helper executes seamlessly."""
    outcome = run_trial(
        sandbox=local_sandbox,
        workdir=local_sandbox.workspace,
        max_steps=2,
    )
    # Default policy fails on step 0, fixes on step 1 -> consolidates!
    assert outcome.status == "consolidated"
    assert outcome.steps_taken == 2
    assert outcome.final_exit_code == 0
    assert outcome.sleep_report is not None


# ---------------------------------------------------------------------------
# 6. Docker Sandbox Integration Test (if Docker daemon responsive)
# ---------------------------------------------------------------------------


def test_docker_sandbox_runner_orchestrator_integration(in_memory_lora_manager: LoRAManager) -> None:
    """Verify DockerSandboxRunner integration through GenesisOrchestrator using client container protocol."""
    from unittest.mock import MagicMock

    class MockExecResult:
        def __init__(self, exit_code: int = 0, stdout: bytes = b"", stderr: bytes = b""):
            self.exit_code = exit_code
            self.output = (stdout, stderr)

    mock_container = MagicMock()
    mock_container.id = "mock_genesis_docker_container"

    clean_diff_output = (
        b"diff --git a/calc.py b/calc.py\n"
        b"--- a/calc.py\n"
        b"+++ b/calc.py\n"
        b"@@ -1,2 +1,2 @@\n"
        b" def add(a, b):\n"
        b"-    return a + b + 1\n"
        b"+    return a + b\n"
    )

    def mock_exec_run(cmd: Any, **kwargs: Any) -> MockExecResult:
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        if "git diff" in cmd_str:
            return MockExecResult(exit_code=0, stdout=clean_diff_output)
        elif "with open('calc.py'" in cmd_str:
            return MockExecResult(exit_code=0, stdout=b"ALL TESTS PASSED\n")
        elif "python3 test_calc.py" in cmd_str:
            return MockExecResult(exit_code=1, stderr=b"AssertionError: assert 5 == 4\n")
        return MockExecResult(exit_code=0, stdout=b"OK\n")

    mock_container.exec_run.side_effect = mock_exec_run
    mock_container.put_archive.return_value = True

    mock_client = MagicMock()
    mock_client.containers.run.return_value = mock_container

    docker_runner = DockerSandboxRunner(
        image="python:3.11-slim",
        workspace="/home/trials",
        client=mock_client,
    )
    docker_runner.start()

    regulator = Homeostasis(dopamine=0.5)
    orchestrator = GenesisOrchestrator(
        sandbox=docker_runner,
        regulator=regulator,
        lora_manager=in_memory_lora_manager,
        workdir="/home/trials",
    )

    outcome = orchestrator.run_trial(
        query_embedding=[1.0] + [0.0] * 15,
        max_steps=2,
    )

    assert outcome.status == "consolidated"
    assert outcome.steps_taken == 2
    assert outcome.final_exit_code == 0
    assert outcome.slot_locked is True
    assert outcome.sleep_report is not None
    assert "-    return a + b + 1" in outcome.sleep_report["diff"]
    assert "+    return a + b" in outcome.sleep_report["diff"]
    assert outcome.sleep_report["tune_result"]["epochs"] == 3


@pytest.mark.skipif(not is_docker_with_git_available(), reason="Docker image with git preinstalled not available")
def test_docker_sandbox_e2e_integration() -> None:
    """Verify full end-to-end cycle inside a real Docker container if git image is available."""
    docker_runner = DockerSandboxRunner(
        image="python:3.11-slim",
        workspace="/home/trials",
    )
    docker_runner.start()
    try:
        regulator = Homeostasis(dopamine=0.5)
        manager = LoRAManager(db_path=":memory:", use_mock=True)
        manager.init_slot("slot_docker", centroid_vector=[1.0] + [0.0] * 15)

        orchestrator = GenesisOrchestrator(
            sandbox=docker_runner,
            regulator=regulator,
            lora_manager=manager,
            workdir="/home/trials",
        )

        outcome = orchestrator.run_trial(
            query_embedding=[1.0] + [0.0] * 15,
            max_steps=2,
        )

        assert outcome.status == "consolidated"
        assert outcome.steps_taken == 2
        assert outcome.final_exit_code == 0
        assert outcome.slot_locked is True
        assert outcome.sleep_report is not None
        assert "-    return a + b + 1" in outcome.sleep_report["diff"]
        assert "+    return a + b" in outcome.sleep_report["diff"]
    finally:
        docker_runner.stop()

