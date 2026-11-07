"""Unit tests for the Memory Network LoRA slot manager (engine/lora_manager.py)."""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import pytest

from engine.lora_manager import (
    LoRAManager,
    MockBaseModel,
    MockLoraConfig,
    MockPeftModel,
    PEFT_AVAILABLE,
    TORCH_AVAILABLE,
)
from interfaces import LoRAManagerProtocol, SlotRecord


def test_protocol_conformance() -> None:
    """Verify that LoRAManager conforms to LoRAManagerProtocol."""
    manager = LoRAManager(use_mock=True)
    assert isinstance(manager, LoRAManagerProtocol)
    manager.close()


def test_sqlite_persistence_in_memory() -> None:
    """Verify in-memory SQLite table operations and CRUD logic."""
    with LoRAManager(use_mock=True, db_path=":memory:") as manager:
        record = manager.init_slot(
            slot_id="slot_alpha",
            centroid_vector=[0.1, 0.2, 0.3],
            salience=0.6,
            is_locked=False,
            adapter_path="/path/to/alpha",
        )
        assert record.slot_id == "slot_alpha"
        assert record.centroid_vector == [0.1, 0.2, 0.3]
        assert record.salience == 0.6
        assert record.is_locked is False
        assert record.adapter_path == "/path/to/alpha"

        fetched = manager.get_slot("slot_alpha")
        assert fetched is not None
        assert fetched.slot_id == "slot_alpha"
        assert fetched.centroid_vector == [0.1, 0.2, 0.3]

        slots = manager.list_slots()
        assert len(slots) == 1
        assert slots[0].slot_id == "slot_alpha"

        # Duplicate init raises ValueError
        with pytest.raises(ValueError, match="already exists"):
            manager.init_slot("slot_alpha", [0.1, 0.2, 0.3])


def test_sqlite_persistence_file_backed() -> None:
    """Verify that slot metadata persists durably across connection re-opens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = os.path.join(tmpdir, "lora_index.db")

        # Open manager, create slots and close
        manager1 = LoRAManager(use_mock=True, db_path=db_file)
        manager1.init_slot("slot_persist_1", [1.0, 0.0], salience=0.8, is_locked=True)
        manager1.init_slot("slot_persist_2", [0.0, 1.0], salience=0.3, is_locked=False)
        manager1.close()

        # Check raw SQLite table directly
        raw_conn = sqlite3.connect(db_file)
        cursor = raw_conn.cursor()
        cursor.execute("SELECT slot_id, salience, is_locked FROM lora_slots ORDER BY slot_id")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == ("slot_persist_1", 0.8, 1)
        assert rows[1] == ("slot_persist_2", 0.3, 0)
        raw_conn.close()

        # Reopen manager with the same DB file and verify state
        manager2 = LoRAManager(use_mock=True, db_path=db_file)
        s1 = manager2.get_slot("slot_persist_1")
        assert s1 is not None
        assert s1.centroid_vector == [1.0, 0.0]
        assert s1.salience == 0.8
        assert s1.is_locked is True

        all_slots = manager2.list_slots()
        assert len(all_slots) == 2
        manager2.close()


def test_lora_config_r8_alpha16() -> None:
    """Verify that initialized slots use LoraConfig with r=8 and lora_alpha=16."""
    with LoRAManager(use_mock=True, lora_r=8, lora_alpha=16) as manager:
        manager.init_slot("slot_lora", [0.5, 0.5])
        assert manager.peft_model is not None
        config = manager.peft_model.peft_config["slot_lora"]
        assert config.r == 8
        assert config.lora_alpha == 16


def test_custom_lora_config_parameters() -> None:
    """Verify custom rank and alpha parameters propagate to PEFT config."""
    with LoRAManager(use_mock=True, lora_r=16, lora_alpha=32) as manager:
        manager.init_slot("slot_custom", [0.1, 0.9])
        config = manager.peft_model.peft_config["slot_custom"]
        assert config.r == 16
        assert config.lora_alpha == 32


def test_cosine_similarity_edge_cases() -> None:
    """Verify cosine similarity calculation under various geometric conditions."""
    calc = LoRAManager.cosine_similarity

    # Identical vectors
    assert math.isclose(calc([1.0, 0.0], [1.0, 0.0]), 1.0)
    assert math.isclose(calc([2.0, 3.0], [4.0, 6.0]), 1.0)

    # Orthogonal vectors
    assert math.isclose(calc([1.0, 0.0], [0.0, 1.0]), 0.0)

    # Opposite vectors
    assert math.isclose(calc([1.0, 0.0], [-1.0, 0.0]), -1.0)

    # Zero vectors
    assert calc([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert calc([0.0, 0.0], [0.0, 0.0]) == 0.0

    # Dimension mismatch
    with pytest.raises(ValueError, match="dimensions do not match"):
        calc([1.0, 2.0], [1.0, 2.0, 3.0])


def test_route_by_similarity() -> None:
    """Verify similarity routing matches the closest centroid accurately."""
    with LoRAManager(use_mock=True) as manager:
        # Empty index routing
        slot, score = manager.route_by_similarity([1.0, 0.0])
        assert slot is None
        assert score == 0.0

        # Register distinct centroids
        manager.init_slot("math_slot", [1.0, 0.0, 0.0], salience=0.7)
        manager.init_slot("code_slot", [0.0, 1.0, 0.0], salience=0.8)
        manager.init_slot("text_slot", [0.0, 0.0, 1.0], salience=0.5)

        # Query close to math
        routed_math, score_math = manager.route_by_similarity([0.9, 0.1, 0.0])
        assert routed_math is not None
        assert routed_math.slot_id == "math_slot"
        assert score_math > 0.9

        # Query close to code
        routed_code, score_code = manager.route_by_similarity([0.05, 0.95, 0.1])
        assert routed_code is not None
        assert routed_code.slot_id == "code_slot"
        assert score_code > 0.9

        # Invalid empty query vector
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.route_by_similarity([])


def test_hot_swap_adapter() -> None:
    """Verify hot-swapping switches the active PEFT adapter and active_slot_id."""
    with LoRAManager(use_mock=True) as manager:
        manager.init_slot("slot_1", [1.0, 0.0])
        manager.init_slot("slot_2", [0.0, 1.0])

        assert manager.active_slot_id is None

        # Swap to slot_1
        res1 = manager.hot_swap_adapter("slot_1")
        assert res1 == "slot_1"
        assert manager.active_slot_id == "slot_1"
        if hasattr(manager.peft_model, "active_adapter"):
            assert manager.peft_model.active_adapter == "slot_1"

        # Swap to slot_2
        res2 = manager.hot_swap_adapter("slot_2")
        assert res2 == "slot_2"
        assert manager.active_slot_id == "slot_2"
        if hasattr(manager.peft_model, "active_adapter"):
            assert manager.peft_model.active_adapter == "slot_2"

        # Hot-swap to nonexistent slot
        with pytest.raises(KeyError, match="not found"):
            manager.hot_swap_adapter("nonexistent")


def test_slot_locking_and_unlocking() -> None:
    """Verify slot locking and unlocking correctly modifies SQLite is_locked state."""
    with LoRAManager(use_mock=True) as manager:
        manager.init_slot("lockable_slot", [0.5, 0.5], is_locked=False)
        assert manager.get_slot("lockable_slot").is_locked is False

        # Lock slot
        manager.lock_slot("lockable_slot")
        assert manager.get_slot("lockable_slot").is_locked is True

        # Unlock slot
        manager.unlock_slot("lockable_slot")
        assert manager.get_slot("lockable_slot").is_locked is False

        # Nonexistent slot
        with pytest.raises(KeyError):
            manager.lock_slot("missing")
        with pytest.raises(KeyError):
            manager.unlock_slot("missing")


def test_update_centroid_ema_and_salience_clamping() -> None:
    """Verify EMA updates to centroid and clamping of salience in [0.0, 1.0]."""
    with LoRAManager(use_mock=True) as manager:
        manager.init_slot("ema_slot", [1.0, 0.0], salience=0.5, is_locked=False)

        # EMA update with learning_rate=0.2:
        # new_centroid = 0.8 * [1.0, 0.0] + 0.2 * [0.0, 1.0] = [0.8, 0.2]
        updated = manager.update_centroid(
            "ema_slot",
            [0.0, 1.0],
            learning_rate=0.2,
            salience_delta=0.2,
        )
        assert len(updated.centroid_vector) == 2
        assert math.isclose(updated.centroid_vector[0], 0.8)
        assert math.isclose(updated.centroid_vector[1], 0.2)
        assert math.isclose(updated.salience, 0.7)

        # Upper salience clamping
        clamped_up = manager.update_centroid("ema_slot", [0.8, 0.2], salience_delta=0.5)
        assert math.isclose(clamped_up.salience, 1.0)

        # Lower salience clamping
        clamped_down = manager.update_centroid("ema_slot", [0.8, 0.2], salience_delta=-2.0)
        assert math.isclose(clamped_down.salience, 0.0)

        # Refusal on locked slot
        manager.lock_slot("ema_slot")
        with pytest.raises(ValueError, match="locked against updates"):
            manager.update_centroid("ema_slot", [0.5, 0.5])

        # Allowed overwrite with force=True
        forced = manager.update_centroid("ema_slot", [0.5, 0.5], force=True)
        assert forced is not None

        # Dimension mismatch
        manager.unlock_slot("ema_slot")
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            manager.update_centroid("ema_slot", [0.5, 0.5, 0.5])


def test_frozen_base_model() -> None:
    """Verify that the base model parameters are frozen (requires_grad is False)."""
    with LoRAManager(use_mock=True) as manager:
        if TORCH_AVAILABLE and manager.peft_model is not None and hasattr(manager.peft_model, "named_parameters"):
            for name, param in manager.peft_model.named_parameters():
                if "lora" not in name:
                    assert param.requires_grad is False


def test_delete_slot() -> None:
    """Verify deleting a slot removes it from the index and unsets active_slot_id."""
    with LoRAManager(use_mock=True) as manager:
        manager.init_slot("slot_del", [1.0, 2.0])
        manager.hot_swap_adapter("slot_del")
        assert manager.active_slot_id == "slot_del"

        deleted = manager.delete_slot("slot_del")
        assert deleted is True
        assert manager.get_slot("slot_del") is None
        assert manager.active_slot_id is None

        # Deleting nonexistent slot returns False
        assert manager.delete_slot("slot_del") is False
