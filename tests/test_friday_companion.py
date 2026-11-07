"""Comprehensive Unit & Integration Test Suite for F.R.I.D.A.Y. Companion & Teacher Bootcamp."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from engine.lora_manager import LoRAManager
from engine.remote_colab import ColabA100Bridge
from friday import FridayCompanion
from interfaces import TrainingPair
from sleep.curriculum import (
    CURRICULUM_SLOTS,
    embed_text,
    init_episodic_memories_table,
    run_teacher_bootcamp,
)


@pytest.fixture
def temp_brain():
    """Create a temporary directory for isolated brain persistence during tests."""
    tmp_dir = tempfile.mkdtemp(prefix="friday_test_brain_")
    db_path = os.path.join(tmp_dir, "test_friday_memory.db")
    adapters_dir = os.path.join(tmp_dir, "adapters")
    yield db_path, adapters_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestCurriculumAndEmbedder:
    """Tests for deterministic 16D semantic domain embedding and Teacher Bootcamp."""

    def test_deterministic_domain_routing_all_six_domains(self):
        """Verify natural English prompts route with > 0.75 cosine similarity to exact matching slot."""
        test_cases = [
            ("hey friday how are you feeling today my friend?", "slot_companion_eq", 0),
            ("how do your hippocampus lora slots and dopamine regulator work?", "slot_self_architecture", 1),
            ("help me fix this python off-by-one pytest bug", "slot_python_mastery", 2),
            ("check git diff and docker container bash logs", "slot_systems_linux", 3),
            ("how does 4-bit nf4 peft lora fine-tuning work on a100?", "slot_ai_engineering", 4),
            ("i am stuck on a hard problem, how should we break it down?", "slot_first_principles", 5),
        ]

        centroids = {slot.slot_id: embed_text(slot.centroid_seed_text) for slot in CURRICULUM_SLOTS}

        for query, expected_slot_id, expected_dim in test_cases:
            q_vec = embed_text(query)
            assert len(q_vec) == 16, "Vector must be 16-dimensional"

            # Dominant dimension check
            assert q_vec[expected_dim] == max(q_vec[:6]), (
                f"Query '{query}' expected highest weight on dim {expected_dim}"
            )

            # Cosine similarity check across centroids
            sims = {
                s_id: LoRAManager.cosine_similarity(q_vec, s_vec)
                for s_id, s_vec in centroids.items()
            }
            best_slot = max(sims, key=sims.get)
            best_sim = sims[best_slot]

            assert best_slot == expected_slot_id, (
                f"Query '{query}' routed to {best_slot}, expected {expected_slot_id}"
            )
            assert best_sim > 0.75, (
                f"Cosine similarity {best_sim:0.4f} for {expected_slot_id} fell below 0.75"
            )

    def test_run_teacher_bootcamp_creates_and_locks_six_slots(self, temp_brain):
        """Verify run_teacher_bootcamp trains 3 epochs and permanently locks all 6 slots with salience >= 0.90."""
        db_path, adapters_dir = temp_brain
        report = run_teacher_bootcamp(
            db_path=db_path,
            adapters_dir=adapters_dir,
            epochs=3,
            use_mock=True,
        )

        assert report["status"] == "completed"
        assert report["total_slots"] == 6

        # Inspect SQLite database directly
        with LoRAManager(db_path=db_path, adapters_dir=adapters_dir, use_mock=True) as mgr:
            slots = mgr.list_slots()
            assert len(slots) == 6

            for slot in slots:
                assert slot.is_locked is True, f"Slot {slot.slot_id} must be locked"
                assert slot.salience >= 0.90, f"Slot {slot.salience} must be >= 0.90"
                assert slot.slot_id in report["slots"]
                assert len(report["slots"][slot.slot_id]["losses"]) == 3

            # Verify episodic memories were persisted
            cursor = mgr.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM episodic_memories")
            count = cursor.fetchone()[0]
            assert count >= 12, "Must contain at least 12 distilled teacher lessons"


class TestFridayCompanionCore:
    """Tests for FridayCompanion initialization, chat turns, HUD, and slash commands."""

    def test_auto_bootcamp_on_new_companion(self, temp_brain):
        """A new FridayCompanion with uninitialized DB automatically runs bootcamp on creation."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        slots = companion.lora_manager.list_slots()
        assert len(slots) == 6
        assert companion.lora_manager.active_slot_id == "slot_companion_eq"

    def test_conversational_chat_and_hud_metadata(self, temp_brain):
        """Verify companion responds warmly, addresses Divyam, and updates live chemical sliders."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        turn = companion.chat("Hey Friday, how are you feeling today my friend?")
        assert "Divyam" in turn["response"]
        assert turn["routed_slot"] == "slot_companion_eq"
        assert turn["similarity"] > 0.75
        assert "dopamine" in turn["mood"]
        assert "cortisol" in turn["mood"]
        assert "adenosine" in turn["mood"]

        hud = companion.render_hud(turn["routed_slot"], turn["similarity"])
        assert "F.R.I.D.A.Y. Mind Status" in hud
        assert "slot_companion_eq" in hud

    def test_slash_commands(self, temp_brain):
        """Verify slash commands (/status, /slots, /teach, /sleep, /run, /colab, /help)."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        # /status
        status_turn = companion.chat("/status")
        assert "Active LoRA Slot:" in status_turn["response"]

        # /slots
        slots_turn = companion.chat("/slots")
        assert "slot_companion_eq" in slots_turn["response"]
        assert "slot_self_architecture" in slots_turn["response"]

        # /teach <topic> :: <insight>
        teach_turn = companion.chat("/teach redis caching :: Always configure key expiry TTLs.")
        assert "Insight consolidated into memory" in teach_turn["response"]
        assert companion.regulator.dopamine > 0.20

        # /sleep consolidation
        companion.regulator.adenosine = 0.85
        companion.regulator.cortisol = 0.50
        sleep_turn = companion.chat("/sleep")
        assert "Sleep consolidation completed" in sleep_turn["response"]
        assert companion.regulator.adenosine == 0.0
        assert companion.regulator.cortisol < 0.50

        # /run
        run_turn = companion.chat("/run echo 'genesis_online'")
        assert "genesis_online" in run_turn["response"]
        assert "SUCCESS" in run_turn["response"]

        # /colab status
        colab_turn = companion.chat("/colab")
        assert "Colab Pro A100 Bridge" in colab_turn["response"]

        # /help
        help_turn = companion.chat("/help")
        assert "F.R.I.D.A.Y. Companion Commands:" in help_turn["response"]


class TestLocalComputerActionsAndSpeed:
    """Tests for natural-language everyday computer actions and sub-50ms execution speed."""

    def test_sub_50ms_chat_turn_latency(self, temp_brain):
        """Verify chat processing is completely local, non-blocking, and executes in < 50ms."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        # Warm-up turn
        companion.chat("hello")

        # Measure 5 consecutive conversational turns
        latencies_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            companion.chat("check python code off-by-one bug")
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        assert avg_latency < 50.0, f"Average turn latency was {avg_latency:0.2f}ms (must be < 50ms)"

    def test_system_stats_check(self, temp_brain):
        """Verify companion can inspect live RAM, CPU load, and disk space in natural English."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        turn = companion.chat("check my ram and cpu status")
        assert "RAM:" in turn["response"]
        assert "CPU Load:" in turn["response"]
        assert "Disk (Root):" in turn["response"]
        assert "Divyam" in turn["response"]

    def test_file_listing_and_search(self, temp_brain):
        """Verify companion can list workspace files and search by pattern."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        # File listing
        list_turn = companion.chat("what files are in this folder")
        assert "Directories:" in list_turn["response"]
        assert "Files:" in list_turn["response"]

        # Search
        find_turn = companion.chat("find friday.py")
        assert "friday.py" in find_turn["response"]

    def test_remember_and_recall_personal_facts(self, temp_brain):
        """Verify companion stores facts into episodic_memories and recalls them upon request."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        store_turn = companion.chat("remember that my preferred model is Qwen-27B on A100")
        assert "Locked into hippocampus memory" in store_turn["response"]
        assert "Qwen-27B" in store_turn["response"]

        recall_turn = companion.chat("what do you remember")
        assert "Qwen-27B" in recall_turn["response"]

    def test_natural_git_status(self, temp_brain):
        """Verify companion inspects git repository changes and commits."""
        db_path, adapters_dir = temp_brain
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            use_mock=True,
            user_name="Divyam",
        )

        git_turn = companion.chat("check git status")
        assert "git workspace status" in git_turn["response"]
        assert "Changes:" in git_turn["response"]


class TestColabA100HeavyComputeOffload:
    """Tests for Colab Pro A100 heavy fine-tuning and patch generation offload."""

    def test_colab_offload_bootcamp_and_sleep(self, temp_brain):
        """Verify heavy 3-epoch backward passes are dispatched to ColabA100Bridge while saving locally."""
        db_path, adapters_dir = temp_brain

        mock_caller = MagicMock()
        mock_caller.side_effect = lambda tool, args: {
            "list_runtimes": {"runtimes": [{"name": "runtimes/colab-a100", "runtimeSpec": {"accelerator": "A100"}}]},
            "create_session": {"session": {"name": "runtimes/colab-a100/sessions/genesis-cortex"}},
            "execute_code": {
                "stdout": '__GENESIS_TUNE__ {"status": "completed", "losses": [0.45, 0.22, 0.11]}'
            },
        }.get(tool, {})

        remote_bridge = ColabA100Bridge(mcp_caller=mock_caller)

        # Run bootcamp with remote bridge
        report = run_teacher_bootcamp(
            db_path=db_path,
            adapters_dir=adapters_dir,
            epochs=3,
            remote_bridge=remote_bridge,
            use_mock=True,
        )

        assert report["remote_offloaded_count"] == 6
        assert report["status"] == "completed"

        # Initialize companion with remote bridge attached
        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            remote_bridge=remote_bridge,
            use_mock=True,
            user_name="Divyam",
        )

        # Test /sleep consolidation offload
        sleep_turn = companion.chat("/sleep")
        assert sleep_turn["remote_offloaded"] is True
        assert "Colab A100" in sleep_turn["response"]

    def test_colab_offline_fallback_to_local(self, temp_brain):
        """When Colab A100 raises an error or is Unauthorized, companion transparently falls back locally."""
        db_path, adapters_dir = temp_brain

        failing_caller = MagicMock(side_effect=PermissionError("Colab MCP Unauthorized: 401"))
        remote_bridge = ColabA100Bridge(mcp_caller=failing_caller)

        companion = FridayCompanion(
            db_path=db_path,
            adapters_dir=adapters_dir,
            remote_bridge=remote_bridge,
            use_mock=True,
            user_name="Divyam",
        )

        # Conversational message should never crash; fallback to local synthesis
        turn = companion.chat("help me fix this python off-by-one bug")
        assert turn["routed_slot"] == "slot_python_mastery"
        assert "Divyam" in turn["response"]
        assert turn["remote_offloaded"] is False
