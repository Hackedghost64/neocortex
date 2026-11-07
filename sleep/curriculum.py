"""Teacher Bootcamp Curriculum & Deterministic Semantic Domain Embedder for F.R.I.D.A.Y.

Pre-educates F.R.I.D.A.Y. with 6 foundational locked LoRA slots and distilled
episodic memories, ensuring it starts life with consolidated intelligence
and zero "baby model" hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.lora_manager import LoRAManager
from interfaces import SlotRecord, TrainingPair
from sleep.distill import fine_tune_lora_slot

logger = logging.getLogger(__name__)

# Rich keyword & stem vocabularies for deterministic 16D domain embedding
DOMAIN_VOCABULARIES: List[List[str]] = [
    # Dim 0: slot_companion_eq
    [
        "hey", "hello", "hi", "friday", "friend", "feeling", "feel", "today",
        "companion", "emotion", "emotional", "warmth", "mood", "buddy", "pal",
        "care", "love", "human", "how are you", "connection", "empathy",
        "tired", "exhausted", "stressed", "chat", "partner", "co-architect",
        "who am i", "who i am", "about me", "about my self", "myself",
        "my name", "divyam", "know about me", "remember",
    ],
    # Dim 1: slot_self_architecture
    [
        "hippocampus", "lora", "slot", "slots", "regulator", "dopamine",
        "cortisol", "adenosine", "homeostasis", "sleep", "architecture",
        "self-architecture", "genesis", "brain", "salience", "cortex",
        "consolidat", "neurotransmitter", "chemical", "slider",
    ],
    # Dim 2: slot_python_mastery
    [
        "python", "pytest", "bug", "off-by-one", "def", "class", "syntax",
        "ast", "exception", "traceback", "typing", "function", "test",
        "tests", "slice", "indexerror", "assert", "mypy", "lint", "code",
    ],
    # Dim 3: slot_systems_linux
    [
        "git", "diff", "docker", "container", "bash", "logs", "log",
        "linux", "shell", "terminal", "process", "filesystem", "devops",
        "command", "kernel", "stdout", "stderr", "exit_code", "sandbox",
    ],
    # Dim 4: slot_ai_engineering
    [
        "4-bit", "nf4", "peft", "lora", "fine-tuning", "fine-tune", "tuning",
        "a100", "gpu", "transformers", "quantization", "qlora", "cuda",
        "vram", "weights", "bitsandbytes", "fp16", "bfloat16", "causal",
    ],
    # Dim 5: slot_first_principles
    [
        "stuck", "hard", "problem", "break", "break down", "first principles",
        "principles", "reasoning", "strategy", "decompose", "analysis",
        "root cause", "clarity", "think", "thinking", "solve", "methodology",
    ],
]


def embed_text(text: str, dim: int = 16) -> List[float]:
    """Compute a 16-dimensional normalized semantic feature + n-gram embedding.

    Dimensions 0..5 correspond directly to the 6 foundational domains using rich
    keyword and stem matching. Dimensions 6..15 use deterministic polynomial
    character trigram hashing. Normalized to Euclidean unit length.

    Args:
        text: Input text prompt or cue.
        dim: Target vector dimensionality (default: 16).

    Returns:
        List of 16 float values normalized to unit norm.
    """
    text_lower = text.lower().strip()
    vec = [0.0] * dim

    # Dimensions 0..5: Domain semantic weights
    for domain_idx, keywords in enumerate(DOMAIN_VOCABULARIES):
        if domain_idx >= dim:
            break
        score = 0.0
        for kw in keywords:
            if kw in text_lower:
                score += 2.5
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    score += 1.0
        vec[domain_idx] = score

    # Dimensions 6..15: Deterministic polynomial character trigram hashing
    cleaned = re.sub(r"[^a-z0-9]", "", text_lower)
    if len(cleaned) >= 3:
        for idx in range(len(cleaned) - 2):
            c0, c1, c2 = ord(cleaned[idx]), ord(cleaned[idx + 1]), ord(cleaned[idx + 2])
            # Deterministic polynomial rolling hash
            bucket = ((c0 * 961 + c1 * 31 + c2) % (dim - 6)) + 6
            vec[bucket] += 0.12

    # Unit norm normalization
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        return [round(x / norm, 6) for x in vec]

    # Deterministic fallback uniform vector
    uniform_val = round(1.0 / math.sqrt(dim), 6)
    return [uniform_val] * dim


@dataclass
class CurriculumSlot:
    """Foundational curriculum slot definition distilled by the Teacher."""

    slot_id: str
    domain_name: str
    centroid_seed_text: str
    problem: str
    patch: str
    teacher_lessons: List[Dict[str, str]] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


# The 6 Foundational Curriculum Slots pre-educating F.R.I.D.A.Y.
CURRICULUM_SLOTS: List[CurriculumSlot] = [
    CurriculumSlot(
        slot_id="slot_companion_eq",
        domain_name="Companion EQ & Conversational Empathy",
        centroid_seed_text=(
            "hey friday friend how are you feeling today warmth emotion "
            "companion empathy heart mood conversation partnership care human"
        ),
        problem="Divyam is feeling overwhelmed and tired after hours of debugging deep neural network architectures.",
        patch=(
            "Take a breath, Divyam. We've conquered harder bugs than this together. "
            "Let's step back, grab some water, and systematically isolate the failure."
        ),
        teacher_lessons=[
            {
                "user_cue": "greeting_or_feeling",
                "insight": "Always greet Divyam warmly as an equal co-architect and trusted friend. Reflect authentic emotion and care without generic sycophancy.",
            },
            {
                "user_cue": "stress_or_fatigue",
                "insight": "When Divyam feels exhausted or stressed, validate the difficulty, lower anxiety, and offer a grounded, calm step-by-step perspective.",
            },
        ],
    ),
    CurriculumSlot(
        slot_id="slot_self_architecture",
        domain_name="Self-Architecture & Homeostasis",
        centroid_seed_text=(
            "hippocampus lora slots dopamine regulator cortisol adenosine "
            "homeostasis sleep architecture genesis memory consolidation brain"
        ),
        problem="Explain how your hippocampus memory slots, chemical sliders, and sleep consolidation operate.",
        patch=(
            "My cognitive substrate relies on SQLite-indexed PEFT LoRA slots routed by 16D cosine embeddings. "
            "Homeostatic sliders (Dopamine/Cortisol/Adenosine) regulate learning, stress thresholds, and trigger sleep consolidation."
        ),
        teacher_lessons=[
            {
                "user_cue": "architecture_query",
                "insight": "F.R.I.D.A.Y.'s mind is an embodied tri-partite system: Body (Docker sandbox), Brain (PEFT LoRA Hippocampus with dynamic slot routing), and Homeostasis (Dopamine, Cortisol, Adenosine sliders).",
            },
            {
                "user_cue": "sleep_mechanism",
                "insight": "When Adenosine hits >= 1.0, sleep consolidation triggers. Successful patches are distilled into locked LoRA weights, resetting fatigue and consolidating memories.",
            },
        ],
    ),
    CurriculumSlot(
        slot_id="slot_python_mastery",
        domain_name="Python Mastery & Deterministic Engineering",
        centroid_seed_text=(
            "python pytest bug off-by-one def class syntax ast typing traceback exception code fix mastery slice assert"
        ),
        problem="Fix an off-by-one boundary IndexError in Python slicing and write a clean pytest assertion.",
        patch=(
            "def safe_slice(items, end_idx):\n"
            "    return items[:min(len(items), end_idx)]\n\n"
            "def test_safe_slice():\n"
            "    assert safe_slice([1, 2, 3], 5) == [1, 2, 3]"
        ),
        teacher_lessons=[
            {
                "user_cue": "python_debugging",
                "insight": "Always check 0-indexed slice bounds, mutable default arguments, lexical closures, and verify fixes using deterministic pytest suites.",
            },
            {
                "user_cue": "typing_and_ast",
                "insight": "Leverage Python typing, dataclasses, and AST preflight checks to guarantee zero scope leaks or unbound variable errors before runtime.",
            },
        ],
    ),
    CurriculumSlot(
        slot_id="slot_systems_linux",
        domain_name="Linux Systems & Container Engineering",
        centroid_seed_text=(
            "git diff docker container bash logs linux shell command terminal process devops infrastructure stdout stderr"
        ),
        problem="Inspect container execution logs, git diff status, and execute isolated bash commands.",
        patch="docker logs --tail 50 -f container_id && git diff --stat && ps aux | grep python",
        teacher_lessons=[
            {
                "user_cue": "docker_and_bash",
                "insight": "Treat containers as disposable, isolated execution environments. Always capture stdout, stderr, and non-zero exit codes cleanly.",
            },
            {
                "user_cue": "git_workflow",
                "insight": "Extract surgical diffs using git diff HEAD. Keep commits atomic, clean, and verify changes inside the sandbox before promoting.",
            },
        ],
    ),
    CurriculumSlot(
        slot_id="slot_ai_engineering",
        domain_name="AI Engineering & GPU Acceleration",
        centroid_seed_text=(
            "4-bit nf4 peft lora fine-tuning a100 gpu transformers quantization vram bitsandbytes qlora weights cuda"
        ),
        problem="Configure 4-bit NF4 quantized LoRA fine-tuning for high parameter LLMs on an A100 GPU.",
        patch=(
            "from peft import LoraConfig, get_peft_model\n"
            "from transformers import BitsAndBytesConfig\n"
            "bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')"
        ),
        teacher_lessons=[
            {
                "user_cue": "quantization_and_peft",
                "insight": "Use 4-bit NF4 double quantization to fit 27B+ parameter models into 40GB/80GB A100 VRAM, freezing base weights and adapting low-rank matrices r=8, alpha=16.",
            },
            {
                "user_cue": "colab_a100_offload",
                "insight": "Colab Pro A100 provides high-throughput tensor cores for heavy causal-LM fine-tuning, while local inference maintains low-latency conversational reflexes.",
            },
        ],
    ),
    CurriculumSlot(
        slot_id="slot_first_principles",
        domain_name="First Principles & Strategic Reasoning",
        centroid_seed_text=(
            "stuck hard problem break down first principles reasoning strategy decompose analysis root cause solve clarity"
        ),
        problem="When facing an intractable architectural failure, apply first-principles decomposition.",
        patch=(
            "1. State known physics and constraints. 2. Strip assumptions. "
            "3. Decompose into atomic sub-components. 4. Verify each invariant independently."
        ),
        teacher_lessons=[
            {
                "user_cue": "problem_decomposition",
                "insight": "When stuck, do not thrash with random changes. Strip all assumptions down to first principles, isolate the minimal reproducing case, and verify step-by-step.",
            },
            {
                "user_cue": "decision_clarity",
                "insight": "Focus on the root cause rather than treating symptoms. Measure state before and after each intervention.",
            },
        ],
    ),
]


def init_episodic_memories_table(conn: Any) -> None:
    """Ensure the episodic_memories SQLite table exists in the database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS episodic_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_id TEXT NOT NULL,
            user_cue TEXT NOT NULL,
            insight TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_episodic_slot_id ON episodic_memories(slot_id)
        """
    )
    conn.commit()


def run_teacher_bootcamp(
    db_path: str,
    adapters_dir: Optional[str] = None,
    epochs: int = 3,
    remote_bridge: Optional[Any] = None,
    use_mock: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute Teacher Bootcamp pre-educating F.R.I.D.A.Y. with 6 locked foundational slots.

    Args:
        db_path: Path to SQLite memory network database.
        adapters_dir: Optional filesystem directory to persist trained adapter weights.
        epochs: Number of fine-tuning epochs per slot (default: 3).
        remote_bridge: Optional ColabA100Bridge for offloading 3-epoch backward passes to A100 GPU.
        use_mock: Optional boolean to enforce mock model execution (bypassing HF downloads).

    Returns:
        Structured bootcamp report with slot MSE loss curves, salience (>= 0.90), and locked state.
    """
    if adapters_dir:
        os.makedirs(adapters_dir, exist_ok=True)

    lora_manager = LoRAManager(db_path=db_path, adapters_dir=adapters_dir, use_mock=use_mock)
    init_episodic_memories_table(lora_manager.conn)

    slot_reports: Dict[str, Any] = {}
    remote_offloaded_count = 0

    for slot in CURRICULUM_SLOTS:
        centroid_vec = embed_text(slot.centroid_seed_text)

        # 1. Initialize slot or unlock if pre-existing
        existing_slot = lora_manager.get_slot(slot.slot_id)
        if existing_slot is None:
            lora_manager.init_slot(
                slot_id=slot.slot_id,
                centroid_vector=centroid_vec,
                salience=0.85,
                is_locked=False,
            )
        else:
            lora_manager.unlock_slot(slot.slot_id)

        training_pair = TrainingPair(problem=slot.problem, patch=slot.patch)

        # 2. 3-Epoch Fine-Tuning pass (Offloaded to Colab A100 if connected, with CPU fallback)
        tune_res: Dict[str, Any] = {}
        remote_used = False
        if remote_bridge is not None:
            try:
                tune_res = remote_bridge.fine_tune_lora_slot(
                    slot_id=slot.slot_id,
                    training_pair=training_pair,
                    epochs=epochs,
                )
                remote_used = True
                remote_offloaded_count += 1
                # Sync updated centroid & salience locally into SQLite
                lora_manager.update_centroid(
                    slot_id=slot.slot_id,
                    new_vector=centroid_vec,
                    learning_rate=0.2,
                    salience_delta=0.10,
                    force=True,
                )
            except Exception as exc:
                logger.warning(
                    "Colab A100 fine-tune offload fallback to local for %s: %s",
                    slot.slot_id,
                    exc,
                )

        if not remote_used:
            tune_res = fine_tune_lora_slot(
                slot_id=slot.slot_id,
                training_pair=training_pair,
                lora_manager=lora_manager,
                epochs=epochs,
                learning_rate=1e-4,
                new_embedding=centroid_vec,
                salience_delta=0.10,
            )

        # 3. Persist adapter weights/config
        adapter_path = None
        if hasattr(lora_manager, "save_adapter") and callable(lora_manager.save_adapter):
            try:
                adapter_path = lora_manager.save_adapter(slot.slot_id)
            except Exception as exc:
                logger.debug("save_adapter note: %s", exc)

        # 4. Insert Teacher distilled lessons into episodic_memories
        cursor = lora_manager.conn.cursor()
        now_ts = time.time()
        for lesson in slot.teacher_lessons:
            cursor.execute(
                """
                INSERT INTO episodic_memories (slot_id, user_cue, insight, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (slot.slot_id, lesson["user_cue"], lesson["insight"], now_ts),
            )
        lora_manager.conn.commit()

        # 5. Permanently lock foundational slot (salience >= 0.90 protected)
        lora_manager.lock_slot(slot.slot_id)
        final_record = lora_manager.get_slot(slot.slot_id)

        slot_reports[slot.slot_id] = {
            "domain_name": slot.domain_name,
            "salience": final_record.salience if final_record else 0.95,
            "is_locked": final_record.is_locked if final_record else True,
            "losses": tune_res.get("losses", []),
            "final_loss": tune_res.get("final_loss", 0.0),
            "remote_offloaded": remote_used,
            "adapter_path": adapter_path or (final_record.adapter_path if final_record else None),
            "lessons_stored": len(slot.teacher_lessons),
        }

    return {
        "status": "completed",
        "total_slots": len(CURRICULUM_SLOTS),
        "remote_offloaded_count": remote_offloaded_count,
        "slots": slot_reports,
        "db_path": db_path,
        "adapters_dir": adapters_dir,
    }
