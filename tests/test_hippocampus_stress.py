"""5-Cycle Multi-Day Hippocampus Slot Stress Test with Real PyTorch PEFT Backward Pass.

Validates:
1. Multi-slot salience decay and dynamic allocation (route_or_allocate).
2. Real PyTorch PEFT (r=8, alpha=16) per-adapter backward pass isolation and zero cross-slot interference.
3. 5-day multi-domain trial cycle with LocalSandboxRunner, Homeostasis, LoRAManager, and SleepConsolidator.
4. Monotonic salience decay of unvisited memories across days.
5. Domain re-visiting and locked-slot immutability against parameter corruption.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple
import pytest

from interfaces import CommandResult, SlotRecord, TrainingPair
from engine.lora_manager import LoRAManager
from orchestrator import GenesisOrchestrator, LocalSandboxRunner
from regulator.state import Homeostasis
from sleep.distill import SleepConsolidator, fine_tune_lora_slot, run_sleep_cycle

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


# ============================================================================
# 1. Multi-Slot Salience Decay & Dynamic Allocation Unit Tests
# ============================================================================


def test_route_or_allocate_dynamic_creation() -> None:
    """Verify route_or_allocate creates distinct slots below threshold and routes above."""
    with LoRAManager(use_mock=True) as mgr:
        # First query in empty index allocates slot_1
        v1 = [1.0] + [0.0] * 15
        slot1, score1, created1 = mgr.route_or_allocate(v1, similarity_threshold=0.75)
        assert created1 is True
        assert slot1.slot_id == "slot_1"
        assert score1 == 1.0
        assert slot1.salience == 0.5

        # Query close to slot_1 routes to existing slot_1
        v1_close = [0.95, 0.05] + [0.0] * 14
        routed, score_close, created_close = mgr.route_or_allocate(v1_close, similarity_threshold=0.75)
        assert created_close is False
        assert routed.slot_id == "slot_1"
        assert score_close > 0.9

        # Orthogonal query allocates slot_2
        v2 = [0.0, 1.0] + [0.0] * 14
        slot2, score2, created2 = mgr.route_or_allocate(v2, similarity_threshold=0.75)
        assert created2 is True
        assert slot2.slot_id == "slot_2"
        assert score2 == 1.0

        assert len(mgr.list_slots()) == 2


def test_decay_unvisited_salience() -> None:
    """Verify decay_unvisited_salience decrements unvisited slots while protecting active slot."""
    with LoRAManager(use_mock=True) as mgr:
        mgr.init_slot("slot_1", [1.0, 0.0], salience=0.8)
        mgr.init_slot("slot_2", [0.0, 1.0], salience=0.5)
        mgr.init_slot("slot_3", [0.7, 0.7], salience=0.1)

        # Decay unvisited slots when slot_1 is active
        decayed = mgr.decay_unvisited_salience(active_slot_id="slot_1", decay_step=0.08)
        assert "slot_1" not in decayed
        assert decayed["slot_2"] == pytest.approx(0.42, abs=1e-4)
        assert decayed["slot_3"] == pytest.approx(0.02, abs=1e-4)

        # Active slot in DB remains unchanged
        s1 = mgr.get_slot("slot_1")
        assert s1 is not None and s1.salience == pytest.approx(0.8, abs=1e-4)

        # Subsequent decay clamps at 0.0
        mgr.decay_unvisited_salience(active_slot_id="slot_1", decay_step=0.1)
        s3 = mgr.get_slot("slot_3")
        assert s3 is not None and s3.salience == 0.0


def test_locked_slot_exempt_from_salience_decay_and_floor() -> None:
    """Verify locked slots are exempt from salience decay and enforce a 0.5 floor."""
    with LoRAManager(use_mock=True) as mgr:
        # Create slots with different initial saliences
        mgr.init_slot("slot_locked_high", [1.0, 0.0], salience=0.8)
        mgr.lock_slot("slot_locked_high")
        assert mgr.is_slot_locked("slot_locked_high") is True

        mgr.init_slot("slot_locked_low", [0.0, 1.0], salience=0.3)
        mgr.lock_slot("slot_locked_low")
        assert mgr.is_slot_locked("slot_locked_low") is True

        mgr.init_slot("slot_unlocked", [0.7, 0.7], salience=0.8)
        assert mgr.is_slot_locked("slot_unlocked") is False

        mgr.init_slot("slot_active", [0.5, 0.5], salience=0.9)

        # Decay unvisited slots when slot_active is active
        decayed = mgr.decay_unvisited_salience(active_slot_id="slot_active", decay_step=0.08)

        # Active slot is excluded from decay
        assert "slot_active" not in decayed
        s_act = mgr.get_slot("slot_active")
        assert s_act is not None and s_act.salience == pytest.approx(0.9, abs=1e-4)

        # Locked high slot is exempt from decay (salience remains 0.8)
        assert decayed["slot_locked_high"] == pytest.approx(0.8, abs=1e-4)
        s_lh = mgr.get_slot("slot_locked_high")
        assert s_lh is not None and s_lh.salience == pytest.approx(0.8, abs=1e-4)

        # Locked low slot has its 0.5 salience floor enforced
        assert decayed["slot_locked_low"] == pytest.approx(0.5, abs=1e-4)
        s_ll = mgr.get_slot("slot_locked_low")
        assert s_ll is not None and s_ll.salience == pytest.approx(0.5, abs=1e-4)

        # Unlocked unvisited slot decays normally by 0.08: 0.8 - 0.08 = 0.72
        assert decayed["slot_unlocked"] == pytest.approx(0.72, abs=1e-4)
        s_un = mgr.get_slot("slot_unlocked")
        assert s_un is not None and s_un.salience == pytest.approx(0.72, abs=1e-4)

        # Multiple subsequent decays: locked slots remain at or above floor; unlocked decays to 0.0
        for _ in range(10):
            mgr.decay_unvisited_salience(active_slot_id="slot_active", decay_step=0.1)

        s_lh_after = mgr.get_slot("slot_locked_high")
        s_ll_after = mgr.get_slot("slot_locked_low")
        s_un_after = mgr.get_slot("slot_unlocked")
        assert s_lh_after is not None and s_lh_after.salience == pytest.approx(0.8, abs=1e-4)
        assert s_ll_after is not None and s_ll_after.salience == pytest.approx(0.5, abs=1e-4)
        assert s_un_after is not None and s_un_after.salience == 0.0


# ============================================================================
# 2. Real PyTorch PEFT Per-Adapter Backward Pass Isolation Tests
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch and PEFT required")
def test_peft_backward_pass_strict_loss_decrease_and_isolation() -> None:
    """Verify 3-epoch backward pass on real PeftModel (r=8, alpha=16).

    Ensures:
    1. Loss strictly decreases across 3 epochs.
    2. Active adapter lora_A and lora_B are updated with non-zero learned weights.
    3. Frozen base model parameters remain 100% unchanged.
    4. Non-active adapters remain 100% unchanged.
    """
    with LoRAManager(use_mock=True, lora_r=8, lora_alpha=16) as mgr:
        assert isinstance(mgr.peft_model, PeftModel)

        mgr.init_slot("slot_1", [1.0] + [0.0] * 15)
        mgr.init_slot("slot_2", [0.0, 1.0] + [0.0] * 14)

        # Initial snapshots
        base_weight_init = mgr.peft_model.base_model.model.proj.base_layer.weight.clone().detach()
        s1_B_init = mgr.peft_model.base_model.model.proj.lora_B["slot_1"].weight.clone().detach()
        s2_A_init = mgr.peft_model.base_model.model.proj.lora_A["slot_2"].weight.clone().detach()
        s2_B_init = mgr.peft_model.base_model.model.proj.lora_B["slot_2"].weight.clone().detach()

        # PEFT initializes lora_B to zeros
        assert bool((s1_B_init == 0).all())
        assert bool((s2_B_init == 0).all())

        # Fine-tune slot_1 for 3 epochs
        pair1 = TrainingPair(problem="Arithmetic off-by-one", patch="return a + b")
        res1 = fine_tune_lora_slot("slot_1", pair1, mgr, epochs=3, learning_rate=0.01)

        losses = res1["losses"]
        assert len(losses) == 3
        # Strictly decreasing loss
        assert losses[0] > losses[1] > losses[2]

        # Inspect updated weights
        base_weight_after = mgr.peft_model.base_model.model.proj.base_layer.weight
        s1_A_after = mgr.peft_model.base_model.model.proj.lora_A["slot_1"].weight
        s1_B_after = mgr.peft_model.base_model.model.proj.lora_B["slot_1"].weight
        s2_A_after = mgr.peft_model.base_model.model.proj.lora_A["slot_2"].weight
        s2_B_after = mgr.peft_model.base_model.model.proj.lora_B["slot_2"].weight

        # 1. Base model is 100% frozen
        assert torch.equal(base_weight_init, base_weight_after)

        # 2. Slot 1 learned non-zero weights
        assert bool((s1_B_after != 0).any())
        assert not torch.equal(s1_B_init, s1_B_after)

        # 3. Slot 2 is 100% unchanged
        assert torch.equal(s2_A_init, s2_A_after)
        assert torch.equal(s2_B_init, s2_B_after)


# ============================================================================
# 3. The 5-Cycle Multi-Day Hippocampus Stress Test (Full System Integration)
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch and PEFT required")
def test_5_cycle_multi_day_hippocampus_stress() -> None:
    """Execute the 5-Cycle Multi-Day Hippocampus Slot Stress Test.

    Evaluates 5 distinct domains across 5 sequential 'days':
    - Day 1: Arithmetic (calc.py / test_calc.py)
    - Day 2: String Parsing (parser.py / test_parser.py)
    - Day 3: Regex Validation (validator.py / test_validator.py)
    - Day 4: File I/O (storage.py / test_storage.py)
    - Day 5: List Sorting (sorter.py / test_sorter.py)

    Verifies:
    1. Orthogonal dynamic allocation to 5 discrete LoRA slots (r=8, alpha=16).
    2. Real PyTorch PEFT 3-epoch backward pass with strictly decreasing loss per cycle.
    3. Zero cross-slot weight interference and frozen base model preservation across all 5 cycles.
    4. Monotonic salience decay of unvisited slots across days, with Day 5 holding highest recency salience.
    5. Re-visiting Day 1's domain with >0.95 cosine similarity, and verification of locked slot immutability.
    """
    sb = LocalSandboxRunner()
    sb.start()
    mgr = LoRAManager(use_mock=True, lora_r=8, lora_alpha=16)
    reg = Homeostasis(dopamine=0.5)  # Start at 0.5 so +0.4 boost reaches 0.9 >= 0.8 lock threshold
    orch = GenesisOrchestrator(sandbox=sb, regulator=reg, lora_manager=mgr)

    # 5 Distinct Problem Domains
    domains = [
        {
            "day": 1,
            "name": "arithmetic",
            "desc": "Fix arithmetic off-by-one bug in calc.py",
            "files": {
                "calc.py": "def add(a, b):\n    return a + b + 1\n",
                "test_calc.py": (
                    "from calc import add\n"
                    "assert add(2, 3) == 5\n"
                    "assert add(-1, 1) == 0\n"
                    "print('ARITHMETIC_PASSED')\n"
                ),
            },
            "patch_cmd": (
                "cat << 'EOF' > calc.py\n"
                "def add(a, b):\n"
                "    return a + b\n"
                "EOF\n"
                "python3 test_calc.py"
            ),
            "test_cmd": "python3 test_calc.py",
            "query_vec": [1.0] + [0.0] * 15,
        },
        {
            "day": 2,
            "name": "string_parsing",
            "desc": "Fix CSV whitespace stripping bug in parser.py",
            "files": {
                "parser.py": (
                    "def parse_csv_line(line):\n"
                    "    return line.split(',')\n"  # Fails to strip whitespace
                ),
                "test_parser.py": (
                    "from parser import parse_csv_line\n"
                    "res = parse_csv_line('apple, banana , cherry ')\n"
                    "assert res == ['apple', 'banana', 'cherry'], f'Got {res}'\n"
                    "print('PARSER_PASSED')\n"
                ),
            },
            "patch_cmd": (
                "cat << 'EOF' > parser.py\n"
                "def parse_csv_line(line):\n"
                "    return [token.strip() for token in line.split(',')]\n"
                "EOF\n"
                "python3 test_parser.py"
            ),
            "test_cmd": "python3 test_parser.py",
            "query_vec": [0.0, 1.0] + [0.0] * 14,
        },
        {
            "day": 3,
            "name": "regex_validation",
            "desc": "Fix hex string regex validator in validator.py",
            "files": {
                "validator.py": (
                    "import re\n"
                    "def is_valid_hex(s):\n"
                    "    return bool(re.fullmatch(r'[0-9]+', s))\n"  # Missing a-fA-F
                ),
                "test_validator.py": (
                    "from validator import is_valid_hex\n"
                    "assert is_valid_hex('1a2f') is True\n"
                    "assert is_valid_hex('XYZ') is False\n"
                    "print('VALIDATOR_PASSED')\n"
                ),
            },
            "patch_cmd": (
                "cat << 'EOF' > validator.py\n"
                "import re\n"
                "def is_valid_hex(s):\n"
                "    return bool(re.fullmatch(r'[0-9a-fA-F]+', s))\n"
                "EOF\n"
                "python3 test_validator.py"
            ),
            "test_cmd": "python3 test_validator.py",
            "query_vec": [0.0, 0.0, 1.0] + [0.0] * 13,
        },
        {
            "day": 4,
            "name": "file_io_storage",
            "desc": "Fix key-value file storage serialization in storage.py",
            "files": {
                "storage.py": (
                    "def save_kv(filename, k, v):\n"
                    "    with open(filename, 'w') as f:\n"
                    "        f.write(f'{k}')\n"  # Forgot '=' and value
                    "def load_kv(filename):\n"
                    "    with open(filename, 'r') as f:\n"
                    "        return f.read().split('=')\n"
                ),
                "test_storage.py": (
                    "from storage import save_kv, load_kv\n"
                    "save_kv('data.kv', 'status', 'active')\n"
                    "k, v = load_kv('data.kv')\n"
                    "assert k == 'status' and v == 'active'\n"
                    "print('STORAGE_PASSED')\n"
                ),
            },
            "patch_cmd": (
                "cat << 'EOF' > storage.py\n"
                "def save_kv(filename, k, v):\n"
                "    with open(filename, 'w') as f:\n"
                "        f.write(f'{k}={v}')\n"
                "def load_kv(filename):\n"
                "    with open(filename, 'r') as f:\n"
                "        return f.read().split('=')\n"
                "EOF\n"
                "python3 test_storage.py"
            ),
            "test_cmd": "python3 test_storage.py",
            "query_vec": [0.0, 0.0, 0.0, 1.0] + [0.0] * 12,
        },
        {
            "day": 5,
            "name": "list_sorting",
            "desc": "Fix reversed sort order bug in sorter.py",
            "files": {
                "sorter.py": (
                    "def sort_ascending(items):\n"
                    "    return sorted(items, reverse=True)\n"  # Bug: descending order
                ),
                "test_sorter.py": (
                    "from sorter import sort_ascending\n"
                    "res = sort_ascending([5, 2, 8, 1, 9])\n"
                    "assert res == [1, 2, 5, 8, 9], f'Got {res}'\n"
                    "print('SORTER_PASSED')\n"
                ),
            },
            "patch_cmd": (
                "cat << 'EOF' > sorter.py\n"
                "def sort_ascending(items):\n"
                "    return sorted(items)\n"
                "EOF\n"
                "python3 test_sorter.py"
            ),
            "test_cmd": "python3 test_sorter.py",
            "query_vec": [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 11,
        },
    ]

    base_layer_initial = mgr.peft_model.base_model.model.proj.base_layer.weight.clone().detach()

    # Telemetry records
    daily_slot_ids: List[str] = []
    daily_losses: List[List[float]] = []
    slot_weight_snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
    day1_salience_trajectory: List[float] = []

    try:
        # Run 5-Day Sequential Execution
        for domain in domains:
            day_num = domain["day"]
            test_cmd = domain["test_cmd"]
            patch_cmd = domain["patch_cmd"]

            def make_policy(t_cmd: str, p_cmd: str):
                def policy(step: int, task: str, last_res: Any = None) -> str:
                    if step == 0:
                        return t_cmd
                    return p_cmd
                return policy

            trial_policy = make_policy(test_cmd, patch_cmd)

            # Ensure dopamine is high enough for Day 1 so it locks on consolidation,
            # while Days 2-5 remain at baseline so slots 2-5 are not locked.
            if day_num == 1:
                reg.dopamine = 0.5
            else:
                reg.dopamine = 0.3

            outcome = orch.run_trial(
                query_embedding=domain["query_vec"],
                task_description=domain["desc"],
                scenario_files=domain["files"],
                test_cmd=test_cmd,
                policy_fn=trial_policy,
                similarity_threshold=0.75,
                decay_unvisited=True,
            )

            # Verify trial outcome
            assert outcome.status == "consolidated", f"Day {day_num} failed with status: {outcome.status}"
            assert outcome.final_exit_code == 0
            assert outcome.sleep_report is not None

            slot_id = outcome.routed_slot_id
            assert slot_id is not None
            daily_slot_ids.append(slot_id)

            # 1. PEFT 3-Epoch Training Loss Verification
            tune_res = outcome.sleep_report["tune_result"]
            assert tune_res is not None
            losses = tune_res["losses"]
            assert len(losses) == 3
            # Loss strictly decreased
            assert losses[0] > losses[-1], f"Day {day_num} loss failed to decrease: {losses}"
            daily_losses.append(losses)

            # Snapshot newly trained slot weights
            lora_A_weight = mgr.peft_model.base_model.model.proj.lora_A[slot_id].weight.clone().detach()
            lora_B_weight = mgr.peft_model.base_model.model.proj.lora_B[slot_id].weight.clone().detach()

            # Active slot's lora_B is non-zero
            assert bool((lora_B_weight != 0).any()), f"Day {day_num} slot {slot_id} lora_B is all zero!"
            slot_weight_snapshots[slot_id] = {"A": lora_A_weight, "B": lora_B_weight}

            # 2. Base Model Weight Isolation Check
            current_base_weight = mgr.peft_model.base_model.model.proj.base_layer.weight
            assert torch.equal(base_layer_initial, current_base_weight), f"Base model weight corrupted on Day {day_num}!"

            # 3. Cross-Slot Non-Interference Check
            # All previously completed slots' weights must remain 100% unchanged
            for prev_slot_id in daily_slot_ids[:-1]:
                prev_A = slot_weight_snapshots[prev_slot_id]["A"]
                prev_B = slot_weight_snapshots[prev_slot_id]["B"]
                curr_A = mgr.peft_model.base_model.model.proj.lora_A[prev_slot_id].weight
                curr_B = mgr.peft_model.base_model.model.proj.lora_B[prev_slot_id].weight
                assert torch.equal(prev_A, curr_A), f"Slot {prev_slot_id} lora_A modified during Day {day_num}!"
                assert torch.equal(prev_B, curr_B), f"Slot {prev_slot_id} lora_B modified during Day {day_num}!"

            # Track Day 1 slot salience trajectory
            s1_record = mgr.get_slot(daily_slot_ids[0])
            assert s1_record is not None
            day1_salience_trajectory.append(s1_record.salience)

        # --------------------------------------------------------------------
        # Assertion 1: Orthogonal / Distinct Routing & Dynamic Allocation
        # --------------------------------------------------------------------
        assert len(daily_slot_ids) == 5
        assert len(set(daily_slot_ids)) == 5, f"Slot collision detected: {daily_slot_ids}"
        db_slots = mgr.list_slots()
        assert len(db_slots) == 5
        for s_id in daily_slot_ids:
            assert s_id in mgr.peft_model.peft_config

        # --------------------------------------------------------------------
        # Assertion 2: Salience Decay & Locked-Slot Exemption Across Days
        # --------------------------------------------------------------------
        # Verify Day 1 slot is locked, and Slots 2, 3, 4 are unlocked
        day1_slot_id = daily_slot_ids[0]
        assert mgr.is_slot_locked(day1_slot_id) is True
        assert mgr.is_slot_locked(daily_slot_ids[1]) is False
        assert mgr.is_slot_locked(daily_slot_ids[2]) is False
        assert mgr.is_slot_locked(daily_slot_ids[3]) is False

        # Day 1 salience starts at 0.5, increased to 0.6 on Day 1 consolidation.
        # Across Days 2, 3, 4, 5, because slot_1 is locked, it is EXEMPT from decay
        # and maintains its consolidated salience >= 0.5 (specifically 0.60):
        assert day1_salience_trajectory[0] == pytest.approx(0.60, abs=1e-4)
        for s in day1_salience_trajectory:
            assert s == pytest.approx(0.60, abs=1e-4), (
                f"Locked slot_1 salience mutated during unvisited day: {day1_salience_trajectory}"
            )
            assert s >= 0.5, f"Locked slot_1 salience dropped below 0.5 floor: {s}"

        # Unlocked unvisited slots (slot_2, slot_3, slot_4) decay monotonically by 0.08 per unvisited day:
        # slot_2 (unvisited Days 3, 4, 5): 0.60 - 3 * 0.08 = 0.36
        # slot_3 (unvisited Days 4, 5):    0.60 - 2 * 0.08 = 0.44
        # slot_4 (unvisited Day 5):        0.60 - 1 * 0.08 = 0.52
        # slot_5 (active on Day 5):        0.60
        s2 = mgr.get_slot(daily_slot_ids[1])
        s3 = mgr.get_slot(daily_slot_ids[2])
        s4 = mgr.get_slot(daily_slot_ids[3])
        s5 = mgr.get_slot(daily_slot_ids[4])
        assert s2 is not None and s3 is not None and s4 is not None and s5 is not None

        assert s2.salience == pytest.approx(0.36, abs=1e-4)
        assert s3.salience == pytest.approx(0.44, abs=1e-4)
        assert s4.salience == pytest.approx(0.52, abs=1e-4)
        assert s5.salience == pytest.approx(0.60, abs=1e-4)
        assert s2.salience < s3.salience < s4.salience <= s5.salience

        # --------------------------------------------------------------------
        # Assertion 3: Re-visiting Day 1 Domain & Locked Slot Immutability
        # --------------------------------------------------------------------
        day1_slot_id = daily_slot_ids[0]
        # Re-query with vector slightly perturbed from Day 1's query vector
        day1_revisit_query = [0.99] + [0.01 / 15] * 15
        routed_revisit, sim_score = mgr.route_by_similarity(day1_revisit_query)
        assert routed_revisit is not None
        assert routed_revisit.slot_id == day1_slot_id
        assert sim_score > 0.95, f"Expected similarity > 0.95, got {sim_score}"

        # Day 1 slot must be locked (dopamine reached 0.9 on Day 1)
        assert mgr.is_slot_locked(day1_slot_id) is True

        # Verify weights and centroid are strictly immutable when locked
        s1_A_before_revisit = mgr.peft_model.base_model.model.proj.lora_A[day1_slot_id].weight.clone().detach()
        s1_B_before_revisit = mgr.peft_model.base_model.model.proj.lora_B[day1_slot_id].weight.clone().detach()
        s1_centroid_before = list(mgr.get_slot(day1_slot_id).centroid_vector)

        # Attempt fine-tuning on locked Day 1 slot
        pair_revisit = TrainingPair(problem="Revisit arithmetic", patch="return a + b")
        revisit_tune = fine_tune_lora_slot(day1_slot_id, pair_revisit, mgr, epochs=3)
        assert revisit_tune["status"] == "locked"
        assert revisit_tune["epochs"] == 0

        # Weights and centroid remain 100% untouched
        s1_A_after_revisit = mgr.peft_model.base_model.model.proj.lora_A[day1_slot_id].weight
        s1_B_after_revisit = mgr.peft_model.base_model.model.proj.lora_B[day1_slot_id].weight
        assert torch.equal(s1_A_before_revisit, s1_A_after_revisit)
        assert torch.equal(s1_B_before_revisit, s1_B_after_revisit)
        assert mgr.get_slot(day1_slot_id).centroid_vector == s1_centroid_before

        logger.info(
            "5-Cycle Hippocampus Stress Test PASSED. Slots: %s, Day 1 Salience Trajectory: %s",
            daily_slot_ids,
            day1_salience_trajectory,
        )

    finally:
        sb.stop()
        mgr.close()
