"""Memory Network LoRA slot manager conforming to LoRAManagerProtocol."""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from interfaces import LoRAManagerProtocol, SlotRecord

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except (ImportError, OSError):
    LoraConfig = None  # type: ignore[assignment,misc]
    PeftModel = None  # type: ignore[assignment,misc]
    get_peft_model = None  # type: ignore[assignment]
    PEFT_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError):
    AutoModelForCausalLM = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


class MockBaseModel(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Lightweight dummy base model for testing and offline execution."""

    def __init__(self, hidden_dim: int = 16) -> None:
        if TORCH_AVAILABLE:
            super().__init__()
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.hidden_dim = hidden_dim

    if TORCH_AVAILABLE:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)


class MockLoraConfig:
    """Mock configuration mirroring PEFT LoraConfig attributes."""

    def __init__(self, r: int = 8, lora_alpha: int = 16, **kwargs: Any) -> None:
        self.r = r
        self.lora_alpha = lora_alpha
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockPeftModel:
    """Mock PEFT model used when real PEFT/torch is unavailable or in mock mode."""

    def __init__(self, default_config: Optional[Any] = None) -> None:
        self.peft_config: Dict[str, Any] = {}
        self.active_adapter: Optional[str] = None
        self._default_config = default_config

    def add_adapter(self, adapter_name: str, config: Any) -> None:
        """Register a new LoRA adapter configuration."""
        self.peft_config[adapter_name] = config

    def set_adapter(self, adapter_name: str) -> None:
        """Activate the specified LoRA adapter."""
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter '{adapter_name}' not registered in PEFT model.")
        self.active_adapter = adapter_name

    def load_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Mock loading an adapter from path."""
        self.peft_config[adapter_name] = self._default_config
        self.active_adapter = adapter_name


class LoRAManager:
    """Manages frozen base model and discrete LoRA slots indexed in SQLite."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        db_path: str = ":memory:",
        adapters_dir: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        base_model: Optional[Any] = None,
        peft_model: Optional[Any] = None,
        use_mock: Optional[bool] = None,
        target_modules: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.db_path = db_path
        self.adapters_dir = adapters_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.device = device
        self.active_slot_id: Optional[str] = None
        self.is_mock: bool = False

        if self.adapters_dir:
            os.makedirs(self.adapters_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        self._init_models(
            base_model=base_model,
            peft_model=peft_model,
            use_mock=use_mock,
        )

    def _init_db(self) -> None:
        """Initialize the SQLite schema for slot metadata."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS lora_slots (
                slot_id TEXT PRIMARY KEY,
                centroid_vector TEXT NOT NULL,
                salience REAL NOT NULL,
                is_locked INTEGER NOT NULL,
                adapter_path TEXT
            )
            """
        )
        self.conn.commit()

    def _create_lora_config(self, target_modules: Optional[List[str]] = None) -> Any:
        """Create a PEFT LoraConfig with configured r and lora_alpha."""
        targets = target_modules or self.target_modules
        if not targets:
            if self.is_mock or isinstance(self.base_model, MockBaseModel):
                targets = ["proj"]
            elif self.base_model is not None:
                targets = ["q_proj", "v_proj"]
            else:
                targets = ["proj"]

        if PEFT_AVAILABLE and LoraConfig is not None:
            kwargs: Dict[str, Any] = {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
            }
            if targets:
                kwargs["target_modules"] = targets
            return LoraConfig(**kwargs)
        return MockLoraConfig(r=self.lora_r, lora_alpha=self.lora_alpha, target_modules=targets)

    def _freeze_base_model(self) -> None:
        """Freeze all parameters of the base model and set evaluation mode."""
        if self.base_model is not None and hasattr(self.base_model, "parameters"):
            for param in self.base_model.parameters():
                param.requires_grad = False
        if self.base_model is not None and hasattr(self.base_model, "eval"):
            self.base_model.eval()

    def _init_models(
        self,
        base_model: Optional[Any],
        peft_model: Optional[Any],
        use_mock: Optional[bool],
    ) -> None:
        """Set up the base and PEFT models with fallback to lightweight mocks."""
        if peft_model is not None:
            self.base_model = base_model
            self.peft_model = peft_model
            self.is_mock = False
            self._freeze_base_model()
            return

        if base_model is not None:
            self.base_model = base_model
            self._freeze_base_model()
            cfg = self._create_lora_config(self.target_modules or ["proj"])
            if PEFT_AVAILABLE and get_peft_model is not None and TORCH_AVAILABLE and isinstance(base_model, nn.Module):
                self.peft_model = get_peft_model(self.base_model, cfg, adapter_name="default_slot")
            else:
                self.peft_model = MockPeftModel(default_config=cfg)
            self.is_mock = False
            return

        if use_mock is True or self.model_name_or_path in ("mock", "dummy"):
            self._setup_mock_model()
            return

        # Attempt loading real model from pretrained weights
        try:
            if not TRANSFORMERS_AVAILABLE or AutoModelForCausalLM is None or not TORCH_AVAILABLE:
                raise RuntimeError("transformers or torch not available for real model loading.")

            kwargs: Dict[str, Any] = {"local_files_only": True}
            if self.device:
                kwargs["device_map"] = self.device
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
            self._freeze_base_model()
            targets = self.target_modules or ["q_proj", "v_proj"]
            cfg = self._create_lora_config(targets)
            if PEFT_AVAILABLE and get_peft_model is not None:
                self.peft_model = get_peft_model(self.base_model, cfg, adapter_name="default_slot")
            else:
                self.peft_model = MockPeftModel(default_config=cfg)
            self.is_mock = False
        except Exception as exc:
            if use_mock is False:
                raise RuntimeError(
                    f"Failed to load real model '{self.model_name_or_path}': {exc}"
                ) from exc
            logger.info(
                "Offline or weights unavailable for '%s' (%s). Initializing lightweight mock execution.",
                self.model_name_or_path,
                exc,
            )
            self._setup_mock_model()

    def _setup_mock_model(self) -> None:
        """Initialize lightweight mock base model and adapter manager."""
        self.is_mock = True
        self.base_model = MockBaseModel()
        self._freeze_base_model()
        targets = self.target_modules or ["proj"]
        cfg = self._create_lora_config(targets)
        if PEFT_AVAILABLE and get_peft_model is not None and TORCH_AVAILABLE and isinstance(self.base_model, nn.Module):
            self.peft_model = get_peft_model(self.base_model, cfg, adapter_name="base_slot")
        else:
            self.peft_model = MockPeftModel(default_config=cfg)

    def init_slot(
        self,
        slot_id: str,
        centroid_vector: Sequence[float],
        salience: float = 0.5,
        is_locked: bool = False,
        adapter_path: Optional[str] = None,
    ) -> SlotRecord:
        """Initialize a discrete LoRA slot with r=8, lora_alpha=16 and persist metadata."""
        if not slot_id or not slot_id.strip():
            raise ValueError("slot_id cannot be empty.")
        if not centroid_vector:
            raise ValueError("centroid_vector cannot be empty.")

        existing = self.get_slot(slot_id)
        if existing is not None:
            raise ValueError(f"Slot '{slot_id}' already exists in SQLite index.")

        clamped_salience = max(0.0, min(1.0, float(salience)))
        vector_list = [float(x) for x in centroid_vector]

        if adapter_path is None and self.adapters_dir:
            adapter_path = os.path.join(self.adapters_dir, slot_id)

        # Register slot in PEFT model if available
        if self.peft_model is not None and hasattr(self.peft_model, "peft_config"):
            if slot_id not in self.peft_model.peft_config:
                cfg = self._create_lora_config()
                self.peft_model.add_adapter(slot_id, cfg)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO lora_slots (slot_id, centroid_vector, salience, is_locked, adapter_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                slot_id,
                json.dumps(vector_list),
                clamped_salience,
                1 if is_locked else 0,
                adapter_path,
            ),
        )
        self.conn.commit()

        return SlotRecord(
            slot_id=slot_id,
            centroid_vector=vector_list,
            salience=clamped_salience,
            is_locked=bool(is_locked),
            adapter_path=adapter_path,
        )

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Vector dimensions do not match: {len(vec_a)} vs {len(vec_b)}"
            )
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            fa = float(a)
            fb = float(b)
            dot += fa * fb
            norm_a += fa * fa
            norm_b += fb * fb

        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0

        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        sim = dot / denom
        return max(-1.0, min(1.0, float(sim)))

    def route_by_similarity(
        self, query_vector: Sequence[float]
    ) -> Tuple[Optional[SlotRecord], float]:
        """Compute cosine similarity against slot centroids and route to highest-scoring slot."""
        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        slots = self.list_slots()
        if not slots:
            return None, 0.0

        best_slot: Optional[SlotRecord] = None
        best_score = float("-inf")

        for slot in slots:
            score = self.cosine_similarity(query_vector, slot.centroid_vector)
            if score > best_score:
                best_score = score
                best_slot = slot

        return best_slot, best_score if best_slot is not None else 0.0

    def route_or_allocate(
        self,
        query_vector: Sequence[float],
        similarity_threshold: float = 0.75,
        slot_prefix: str = "slot",
    ) -> Tuple[SlotRecord, float, bool]:
        """Route to existing slot by cosine similarity or allocate a new discrete slot.

        Routes via route_by_similarity(query_vector); if no slot exists or
        best_score < similarity_threshold, initializes a new slot (init_slot)
        with centroid_vector=query_vector, returning (slot_record, score, created_new).

        Args:
            query_vector: Query embedding vector.
            similarity_threshold: Cosine similarity cutoff to reuse an existing slot (default: 0.75).
            slot_prefix: Prefix used when generating a new slot_id (default: 'slot').

        Returns:
            Tuple of (slot_record, score, created_new).
        """
        best_slot, best_score = self.route_by_similarity(query_vector)
        if best_slot is not None and best_score >= similarity_threshold:
            return best_slot, best_score, False

        idx = 1
        while self.get_slot(f"{slot_prefix}_{idx}") is not None:
            idx += 1
        new_slot_id = f"{slot_prefix}_{idx}"

        new_slot = self.init_slot(
            slot_id=new_slot_id,
            centroid_vector=query_vector,
            salience=0.5,
        )
        return new_slot, 1.0, True

    def hot_swap_adapter(self, slot_id: str) -> str:
        """Activate the target LoRA adapter on PEFT model and update active_slot_id."""
        slot = self.get_slot(slot_id)
        if slot is None:
            raise KeyError(f"Slot '{slot_id}' not found in SQLite index.")

        if self.peft_model is not None:
            has_adapter = False
            if hasattr(self.peft_model, "peft_config") and slot_id in self.peft_model.peft_config:
                has_adapter = True

            if not has_adapter:
                loaded = False
                if slot.adapter_path and os.path.exists(slot.adapter_path):
                    load_path = slot.adapter_path
                    if not os.path.exists(os.path.join(load_path, "adapter_config.json")):
                        sub_path = os.path.join(load_path, slot_id)
                        if os.path.exists(os.path.join(sub_path, "adapter_config.json")):
                            load_path = sub_path
                    try:
                        self.peft_model.load_adapter(load_path, adapter_name=slot_id)
                        loaded = True
                    except Exception as exc:
                        logger.warning("Could not load adapter from %s: %s", load_path, exc)

                if not loaded:
                    cfg = self._create_lora_config()
                    self.peft_model.add_adapter(slot_id, cfg)

            self.peft_model.set_adapter(slot_id)

        self.active_slot_id = slot_id
        return slot_id

    def save_adapter(self, slot_id: str, save_directory: Optional[str] = None) -> str:
        """Persist trained PEFT LoRA adapter weights or config to disk."""
        slot = self.get_slot(slot_id)
        if slot is None:
            raise KeyError(f"Slot '{slot_id}' not found in SQLite index.")

        target_dir = save_directory or slot.adapter_path
        if not target_dir:
            if self.adapters_dir:
                target_dir = os.path.join(self.adapters_dir, slot_id)
            else:
                target_dir = os.path.join("adapters", slot_id)

        os.makedirs(target_dir, exist_ok=True)

        # Attempt native PEFT save_pretrained
        saved_native = False
        if self.peft_model is not None and hasattr(self.peft_model, "save_pretrained") and callable(self.peft_model.save_pretrained):
            try:
                self.peft_model.save_pretrained(target_dir, selected_adapters=[slot_id])
                saved_native = True
                sub_dir = os.path.join(target_dir, slot_id)
                if os.path.exists(sub_dir) and not os.path.exists(os.path.join(target_dir, "adapter_config.json")):
                    for fname in os.listdir(sub_dir):
                        src_f = os.path.join(sub_dir, fname)
                        dst_f = os.path.join(target_dir, fname)
                        if not os.path.exists(dst_f):
                            try:
                                import shutil
                                shutil.copy2(src_f, dst_f)
                            except Exception:
                                pass
            except Exception as exc:
                logger.debug("PEFT native save_pretrained fallback for %s: %s", slot_id, exc)

        if not saved_native:
            # Deterministic serialization fallback
            cfg_path = os.path.join(target_dir, "adapter_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "r": self.lora_r,
                        "lora_alpha": self.lora_alpha,
                        "slot_id": slot_id,
                        "peft_type": "LORA",
                        "target_modules": self.target_modules or ["proj"],
                    },
                    f,
                    indent=2,
                )

        cursor = self.conn.cursor()
        cursor.execute("UPDATE lora_slots SET adapter_path = ? WHERE slot_id = ?", (target_dir, slot_id))
        self.conn.commit()
        return target_dir

    def lock_slot(self, slot_id: str) -> None:
        """Lock slot in SQLite, protecting it from destructive modification."""
        slot = self.get_slot(slot_id)
        if slot is None:
            raise KeyError(f"Slot '{slot_id}' not found.")
        cursor = self.conn.cursor()
        cursor.execute("UPDATE lora_slots SET is_locked = 1 WHERE slot_id = ?", (slot_id,))
        self.conn.commit()

    def unlock_slot(self, slot_id: str) -> None:
        """Unlock slot in SQLite, permitting centroid updates."""
        slot = self.get_slot(slot_id)
        if slot is None:
            raise KeyError(f"Slot '{slot_id}' not found.")
        cursor = self.conn.cursor()
        cursor.execute("UPDATE lora_slots SET is_locked = 0 WHERE slot_id = ?", (slot_id,))
        self.conn.commit()

    def update_centroid(
        self,
        slot_id: str,
        new_vector: Sequence[float],
        learning_rate: float = 0.2,
        salience_delta: float = 0.0,
        force: bool = False,
    ) -> SlotRecord:
        """Update centroid via EMA and clamp salience to [0.0, 1.0]."""
        slot = self.get_slot(slot_id)
        if slot is None:
            raise KeyError(f"Slot '{slot_id}' not found.")

        if slot.is_locked and not force:
            raise ValueError(f"Slot '{slot_id}' is locked against updates.")

        if len(new_vector) != len(slot.centroid_vector):
            raise ValueError(
                f"Vector dimension mismatch: expected {len(slot.centroid_vector)}, got {len(new_vector)}"
            )

        lr = float(learning_rate)
        updated_centroid = [
            (1.0 - lr) * float(old_val) + lr * float(new_val)
            for old_val, new_val in zip(slot.centroid_vector, new_vector)
        ]

        new_salience = max(0.0, min(1.0, float(slot.salience + salience_delta)))

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE lora_slots
            SET centroid_vector = ?, salience = ?
            WHERE slot_id = ?
            """,
            (json.dumps(updated_centroid), new_salience, slot_id),
        )
        self.conn.commit()

        return SlotRecord(
            slot_id=slot_id,
            centroid_vector=updated_centroid,
            salience=new_salience,
            is_locked=slot.is_locked,
            adapter_path=slot.adapter_path,
        )

    def is_slot_locked(self, slot_id: str) -> bool:
        """Check if slot is marked locked in SQLite index."""
        slot = self.get_slot(slot_id)
        return bool(slot.is_locked) if slot is not None else False

    def is_locked(self, slot_id: str) -> bool:
        """Alias for is_slot_locked."""
        return self.is_slot_locked(slot_id)

    def decay_unvisited_salience(
        self,
        active_slot_id: Optional[str] = None,
        decay_step: float = 0.08,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Decay salience for unvisited LoRA slots in SQLite index.

        Decays salience (clamped to [0.0, 1.0]) in SQLite for all slots where
        slot_id != active_slot_id, returning a dict of {slot_id: new_salience}.

        Args:
            active_slot_id: Identifier of currently active slot to exclude from decay.
            decay_step: Salience decrement per cycle (default: 0.08).

        Returns:
            Dict mapping slot_id to updated salience for all unvisited slots.
        """
        step = float(kwargs.get("decay_Step", decay_step))
        cursor = self.conn.cursor()
        cursor.execute("SELECT slot_id, salience, is_locked FROM lora_slots")
        rows = cursor.fetchall()
        decayed: Dict[str, float] = {}
        for slot_id, salience, is_locked in rows:
            if active_slot_id is not None and slot_id == active_slot_id:
                continue
            if bool(is_locked):
                # Exempt locked slots from decay and enforce a 0.5 salience floor
                new_salience = max(0.5, float(salience))
            else:
                new_salience = max(0.0, min(1.0, float(salience) - step))
            cursor.execute(
                "UPDATE lora_slots SET salience = ? WHERE slot_id = ?",
                (new_salience, slot_id),
            )
            decayed[slot_id] = new_salience
        self.conn.commit()
        return decayed

    def get_slot(self, slot_id: str) -> Optional[SlotRecord]:
        """Fetch a single slot record by slot_id."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT slot_id, centroid_vector, salience, is_locked, adapter_path
            FROM lora_slots
            WHERE slot_id = ?
            """,
            (slot_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return SlotRecord(
            slot_id=row[0],
            centroid_vector=json.loads(row[1]),
            salience=float(row[2]),
            is_locked=bool(row[3]),
            adapter_path=row[4],
        )

    def list_slots(self) -> List[SlotRecord]:
        """List all slot records in the SQLite index."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT slot_id, centroid_vector, salience, is_locked, adapter_path
            FROM lora_slots
            ORDER BY slot_id ASC
            """
        )
        rows = cursor.fetchall()
        return [
            SlotRecord(
                slot_id=r[0],
                centroid_vector=json.loads(r[1]),
                salience=float(r[2]),
                is_locked=bool(r[3]),
                adapter_path=r[4],
            )
            for r in rows
        ]

    def delete_slot(self, slot_id: str) -> bool:
        """Delete a slot record from SQLite index."""
        slot = self.get_slot(slot_id)
        if slot is None:
            return False
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM lora_slots WHERE slot_id = ?", (slot_id,))
        self.conn.commit()
        if self.active_slot_id == slot_id:
            self.active_slot_id = None
        return True

    def close(self) -> None:
        """Close database connection."""
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> LoRAManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
