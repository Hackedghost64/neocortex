"""Unit tests for the Sleep & Consolidation Pipeline."""

from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from interfaces import CommandResult, SlotRecord, TrainingPair
from sleep.distill import (
    SleepConsolidator,
    extract_clean_diff,
    fine_tune_lora_slot,
    run_sleep_cycle,
    synthesize_training_pair,
)


class MockSandbox:
    """Mock sandbox implementing run_command for diff testing."""

    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.last_command: Optional[str] = None
        self.last_workdir: Optional[str] = None

    def run_command(self, command: str, workdir: str = "/home/trials", timeout: int = 60) -> CommandResult:
        self.last_command = command
        self.last_workdir = workdir
        return CommandResult(
            command=command,
            stdout=self.stdout,
            stderr=self.stderr,
            exit_code=self.exit_code,
        )


class MockLoRAManager:
    """Mock memory manager tracking slot records and centroid updates."""

    def __init__(self, slots: Optional[Dict[str, SlotRecord]] = None) -> None:
        self.slots: Dict[str, SlotRecord] = slots or {}
        self.update_centroid_calls: List[Tuple[str, Sequence[float], float]] = []
        self.recorded_steps: List[Tuple[str, int, float]] = []

    def get_slot(self, slot_id: str) -> Optional[SlotRecord]:
        return self.slots.get(slot_id)

    def is_locked(self, slot_id: str) -> bool:
        slot = self.slots.get(slot_id)
        return slot.is_locked if slot is not None else False

    def update_centroid(
        self,
        slot_id: str,
        new_vector: Sequence[float],
        learning_rate: float = 0.2,
        salience_delta: float = 0.0,
    ) -> SlotRecord:
        self.update_centroid_calls.append((slot_id, list(new_vector), learning_rate))
        if slot_id in self.slots:
            self.slots[slot_id].centroid_vector = list(new_vector)
            return self.slots[slot_id]
        new_record = SlotRecord(slot_id=slot_id, centroid_vector=list(new_vector))
        self.slots[slot_id] = new_record
        return new_record

    def record_training_step(self, slot_id: str, epochs: int, lr: float) -> None:
        self.recorded_steps.append((slot_id, epochs, lr))

    def hot_swap_adapter(self, slot_id: str) -> str:
        return f"/adapters/{slot_id}"


class MockRegulator:
    """Mock homeostatic regulator tracking adenosine resets."""

    def __init__(self, adenosine: float = 0.85) -> None:
        self.adenosine = adenosine
        self.reset_called = False
        self.reset_count = 0

    def reset_adenosine(self) -> None:
        self.reset_called = True
        self.reset_count += 1
        self.adenosine = 0.0


# ---------------------------------------------------------------------------
# 1. Diff Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_clean_diff_sandbox_success():
    """Verify clean diff extraction from sandbox runner on success."""
    sample_diff = """diff --git a/solve.py b/solve.py
--- a/solve.py
+++ b/solve.py
@@ -1 +1 @@
-return False
+return True"""
    sandbox = MockSandbox(exit_code=0, stdout=sample_diff)
    diff = extract_clean_diff(sandbox, workdir="/home/trials")
    assert diff == sample_diff.strip()
    assert sandbox.last_command == "git diff HEAD"
    assert sandbox.last_workdir == "/home/trials"


def test_extract_clean_diff_sandbox_failure():
    """Verify non-zero exit code returns empty string (or raises if requested)."""
    sandbox = MockSandbox(exit_code=1, stderr="fatal: not a git repo")
    diff = extract_clean_diff(sandbox)
    assert diff == ""

    with pytest.raises(RuntimeError):
        extract_clean_diff(sandbox, raise_on_error=True)


def test_extract_clean_diff_command_result_object():
    """Verify handling of direct CommandResult objects."""
    res_success = CommandResult(command="git diff HEAD", stdout="diff abc", stderr="", exit_code=0)
    assert extract_clean_diff(res_success) == "diff abc"

    res_fail = CommandResult(command="git diff HEAD", stdout="", stderr="err", exit_code=1)
    assert extract_clean_diff(res_fail) == ""


def test_extract_clean_diff_local_git_repo(tmp_path):
    """Verify diff extraction from an actual local git directory."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        check=True,
    )

    f = tmp_path / "solution.py"
    f.write_text("def solve():\n    return False\n")
    subprocess.run(["git", "add", "solution.py"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    # Clean repo: no diff
    assert extract_clean_diff(str(tmp_path)) == ""

    # Modify file
    f.write_text("def solve():\n    return True\n")
    diff = extract_clean_diff(str(tmp_path))
    assert "+    return True" in diff
    assert "-    return False" in diff


# ---------------------------------------------------------------------------
# 2. Training Pair Synthesis Tests
# ---------------------------------------------------------------------------


def test_synthesize_training_pair_deterministic():
    """Verify deterministic <problem> -> <patch> formatting without teacher."""
    problem = "Fix IndexError in array lookup"
    patch = "--- a/arr.py\n+++ b/arr.py\n- return arr[i]\n+ return arr[i] if i < len(arr) else None"

    pair = synthesize_training_pair(problem, patch, teacher_fn=None)
    assert isinstance(pair, TrainingPair)
    assert pair.problem == problem
    assert pair.patch == patch.strip()
    assert pair.formatted_text == f"<problem>\n{problem}\n</problem>\n<patch>\n{patch.strip()}\n</patch>"


def test_synthesize_training_pair_with_teacher_fn():
    """Verify teacher model distillation callback."""
    problem = "Optimize fibonacci"
    diff = "+ use memoization"

    def mock_teacher(prompt: str) -> str:
        return "+ return memo[n]"

    pair = synthesize_training_pair(problem, diff, teacher_fn=mock_teacher)
    assert pair.problem == problem
    assert pair.patch == "+ return memo[n]"
    assert "<patch>\n+ return memo[n]\n</patch>" in pair.formatted_text


def test_synthesize_training_pair_with_xml_teacher():
    """Verify teacher output containing XML tags is correctly parsed."""
    problem = "Original prompt"
    diff = "+ raw diff"

    def tagged_teacher(prompt: str) -> str:
        return "<problem>Refined problem statement</problem>\n<patch>Cleaned patch contents</patch>"

    pair = synthesize_training_pair(problem, diff, teacher_fn=tagged_teacher)
    assert pair.problem == "Refined problem statement"
    assert pair.patch == "Cleaned patch contents"


# ---------------------------------------------------------------------------
# 3. Fine-Tuning & 3-Epoch Training Pass Tests
# ---------------------------------------------------------------------------


def test_fine_tune_lora_slot_3_epochs_mock():
    """Verify 3-epoch execution in mock/CPU mode with loss progression."""
    slot = SlotRecord(slot_id="slot_alpha", centroid_vector=[0.0] * 8, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_alpha": slot})
    pair = TrainingPair(problem="Add error handling", patch="+ try: ... except: ...")

    result = fine_tune_lora_slot(
        slot_id="slot_alpha",
        training_pair=pair,
        lora_manager=lora_mgr,
        epochs=3,
        learning_rate=1e-4,
    )

    assert result["status"] == "completed"
    assert result["epochs"] == 3
    assert len(result["losses"]) == 3
    # Check that losses follow expected deterministic progression
    assert result["losses"][0] > result["losses"][1] > result["losses"][2]
    assert ("slot_alpha", 3, 1e-4) in lora_mgr.recorded_steps
    assert len(lora_mgr.update_centroid_calls) == 1


def test_fine_tune_lora_slot_3_epochs_pytorch():
    """Verify 3-epoch execution when a live PyTorch model is attached."""
    model = nn.Linear(8, 2)
    # Save initial weights
    initial_weight = model.weight.clone().detach()

    slot = SlotRecord(slot_id="slot_torch", centroid_vector=[0.1] * 8, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_torch": slot})
    # Attach model to lora_mgr
    lora_mgr.model = model

    pair = TrainingPair(problem="Vector task", patch="+ output tensor")

    result = fine_tune_lora_slot(
        slot_id="slot_torch",
        training_pair=pair,
        lora_manager=lora_mgr,
        epochs=3,
        learning_rate=0.05,
    )

    assert result["status"] == "completed"
    assert result["epochs"] == 3
    assert len(result["losses"]) == 3
    # Parameters should have changed due to backward + step
    assert not torch.allclose(model.weight, initial_weight)


# ---------------------------------------------------------------------------
# 4. Locked-Slot Protection Tests
# ---------------------------------------------------------------------------


def test_locked_slot_skips_fine_tuning():
    """Verify locked slot skips weight mutation and does not update centroid."""
    slot = SlotRecord(slot_id="slot_locked", centroid_vector=[1.0, 0.0], is_locked=True)
    lora_mgr = MockLoRAManager(slots={"slot_locked": slot})
    pair = TrainingPair(problem="Locked task", patch="+ safe change")

    result = fine_tune_lora_slot(
        slot_id="slot_locked",
        training_pair=pair,
        lora_manager=lora_mgr,
        epochs=3,
    )

    assert result["status"] == "locked"
    assert result["epochs"] == 0
    assert result["losses"] == []
    assert len(lora_mgr.recorded_steps) == 0
    assert len(lora_mgr.update_centroid_calls) == 0
    # Centroid vector remains untouched
    assert slot.centroid_vector == [1.0, 0.0]


# ---------------------------------------------------------------------------
# 5. Centroid Vector Update Tests
# ---------------------------------------------------------------------------


def test_centroid_update_explicit_embedding():
    """Verify explicit query embedding is passed to update_centroid."""
    slot = SlotRecord(slot_id="slot_embed", centroid_vector=[0.0] * 4, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_embed": slot})
    pair = TrainingPair(problem="Task", patch="+ diff")
    explicit_vec = [0.25, 0.5, 0.75, 1.0]

    result = fine_tune_lora_slot(
        slot_id="slot_embed",
        training_pair=pair,
        lora_manager=lora_mgr,
        epochs=3,
        new_embedding=explicit_vec,
    )

    assert result["status"] == "completed"
    assert len(lora_mgr.update_centroid_calls) == 1
    call_slot_id, call_vec, lr = lora_mgr.update_centroid_calls[0]
    assert call_slot_id == "slot_embed"
    assert call_vec == explicit_vec


def test_centroid_update_derived_embedding():
    """Verify automatic derivation of normalized centroid vector when embedding is None."""
    slot = SlotRecord(slot_id="slot_auto", centroid_vector=[0.0] * 16, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_auto": slot})
    pair = TrainingPair(problem="Automatic embedding task", patch="+ automatic diff")

    result = fine_tune_lora_slot(
        slot_id="slot_auto",
        training_pair=pair,
        lora_manager=lora_mgr,
        epochs=3,
        new_embedding=None,
    )

    assert result["status"] == "completed"
    assert len(lora_mgr.update_centroid_calls) == 1
    call_slot_id, call_vec, _ = lora_mgr.update_centroid_calls[0]
    assert call_slot_id == "slot_auto"
    assert len(call_vec) == 16
    # Vector should be approximately unit length
    magnitude = sum(x * x for x in call_vec) ** 0.5
    assert pytest.approx(magnitude, rel=1e-2) == 1.0


# ---------------------------------------------------------------------------
# 6. End-to-End Sleep Cycle & Reset Adenosine Tests
# ---------------------------------------------------------------------------


def test_run_sleep_cycle_full_consolidation():
    """Verify full sleep cycle extracts diff, synthesizes pair, fine-tunes, and resets adenosine."""
    diff_text = """diff --git a/test.py b/test.py
+ def fixed(): return True"""
    sandbox = MockSandbox(exit_code=0, stdout=diff_text)
    slot = SlotRecord(slot_id="slot_0", centroid_vector=[0.1] * 8, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_0": slot})
    regulator = MockRegulator(adenosine=0.9)

    result = run_sleep_cycle(
        problem_description="Fix test bug",
        sandbox=sandbox,
        lora_manager=lora_mgr,
        regulator=regulator,
        slot_id="slot_0",
        query_embedding=[0.2] * 8,
        epochs=3,
    )

    assert result["status"] == "success"
    assert result["diff"] == diff_text.strip()
    assert result["adenosine_reset"] is True
    assert result["tune_result"]["status"] == "completed"
    assert result["tune_result"]["epochs"] == 3
    assert regulator.reset_called is True
    assert regulator.reset_count == 1
    assert regulator.adenosine == 0.0
    assert len(lora_mgr.update_centroid_calls) == 1


def test_run_sleep_cycle_no_diff_skips_tune():
    """Verify when diff is empty, consolidation is skipped and adenosine is not reset."""
    sandbox = MockSandbox(exit_code=0, stdout="")
    slot = SlotRecord(slot_id="slot_0", centroid_vector=[0.1] * 8, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_0": slot})
    regulator = MockRegulator(adenosine=0.7)

    result = run_sleep_cycle(
        problem_description="No change task",
        sandbox=sandbox,
        lora_manager=lora_mgr,
        regulator=regulator,
        slot_id="slot_0",
    )

    assert result["status"] == "no_diff"
    assert result["adenosine_reset"] is False
    assert regulator.reset_called is False
    assert len(lora_mgr.update_centroid_calls) == 0


def test_sleep_consolidator_class_methods():
    """Verify SleepConsolidator coordinator class API."""
    slot = SlotRecord(slot_id="slot_c", centroid_vector=[0.0] * 4, is_locked=False)
    lora_mgr = MockLoRAManager(slots={"slot_c": slot})
    regulator = MockRegulator(adenosine=0.8)
    consolidator = SleepConsolidator(lora_manager=lora_mgr, regulator=regulator, default_epochs=3)

    # Class diff extraction
    sandbox = MockSandbox(exit_code=0, stdout="+ change")
    diff = consolidator.extract_clean_diff(sandbox)
    assert diff == "+ change"

    # Class pair synthesis
    pair = consolidator.synthesize_training_pair("Problem text", diff)
    assert isinstance(pair, TrainingPair)

    # Class fine tune
    tune_res = consolidator.fine_tune_lora_slot(slot_id="slot_c", training_pair=pair)
    assert tune_res["status"] == "completed"
    assert tune_res["epochs"] == 3

    # Class sleep cycle
    cycle_res = consolidator.run_sleep_cycle(
        problem_description="Cycle task",
        sandbox=sandbox,
        slot_id="slot_c",
    )
    assert cycle_res["status"] == "success"
    assert regulator.reset_called is True
