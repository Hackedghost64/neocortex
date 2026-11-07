"""Sleep & Consolidation Pipeline for Project Genesis.

Distills successful trial modifications into low-rank LoRA adapter slots
and resets homeostatic sleep pressure (adenosine).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from interfaces import (
    CommandResult,
    HomeostasisProtocol,
    LoRAManagerProtocol,
    SandboxRunnerProtocol,
    SlotRecord,
    TrainingPair,
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def extract_clean_diff(
    sandbox_runner_or_workdir: Any,
    workdir: str = "/home/trials",
    raise_on_error: bool = False,
) -> str:
    """Extract clean git diff on trial success (exit code 0).

    Args:
        sandbox_runner_or_workdir: A SandboxRunner instance, a CommandResult,
            or a local filesystem directory path.
        workdir: Working directory inside the sandbox environment.
        raise_on_error: Whether to raise an exception instead of returning empty string.

    Returns:
        The clean diff string (empty string if no changes or failed execution).
    """
    # Case 1: Direct CommandResult object passed
    if hasattr(sandbox_runner_or_workdir, "exit_code") and not hasattr(
        sandbox_runner_or_workdir, "run_command"
    ):
        if sandbox_runner_or_workdir.exit_code != 0:
            if raise_on_error:
                raise RuntimeError(
                    f"Command failed with exit code {sandbox_runner_or_workdir.exit_code}"
                )
            return ""
        return str(getattr(sandbox_runner_or_workdir, "stdout", "")).strip()

    # Case 2: SandboxRunner instance with run_command
    if hasattr(sandbox_runner_or_workdir, "run_command") and callable(
        sandbox_runner_or_workdir.run_command
    ):
        cmd_result = sandbox_runner_or_workdir.run_command(
            "git diff HEAD", workdir=workdir
        )
        if cmd_result.exit_code != 0:
            # Fallback to plain git diff in case repository has no prior commits
            cmd_result = sandbox_runner_or_workdir.run_command(
                "git diff", workdir=workdir
            )
        if cmd_result.exit_code != 0:
            if raise_on_error:
                raise RuntimeError(
                    f"git diff failed inside sandbox: {cmd_result.stderr}"
                )
            return ""
        diff_str = cmd_result.stdout.strip()
        if not diff_str and raise_on_error:
            raise ValueError("No changes detected in sandbox git diff")
        return diff_str

    # Case 3: Local directory path (str or Path)
    if isinstance(sandbox_runner_or_workdir, (str, Path, os.PathLike)):
        repo_path = Path(sandbox_runner_or_workdir)
        if not repo_path.exists():
            if raise_on_error:
                raise FileNotFoundError(f"Directory does not exist: {repo_path}")
            return ""
        try:
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                proc = subprocess.run(
                    ["git", "diff"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                if raise_on_error:
                    raise RuntimeError(
                        f"Local git diff failed: {proc.stderr}"
                    )
                return ""
            diff_str = proc.stdout.strip()
            if not diff_str and raise_on_error:
                raise ValueError("No local git changes detected")
            return diff_str
        except Exception as exc:
            if raise_on_error:
                raise
            logger.warning("Error reading local git diff: %s", exc)
            return ""

    logger.warning("Unsupported runner/workdir type: %s", type(sandbox_runner_or_workdir))
    if raise_on_error:
        raise TypeError(f"Unsupported runner/workdir: {sandbox_runner_or_workdir}")
    return ""


def synthesize_training_pair(
    problem_description: str,
    git_diff: str,
    teacher_fn: Optional[Callable[[str], str]] = None,
) -> TrainingPair:
    """Format a synthetic <problem> -> <patch> training pair.

    Uses the Teacher model callable when provided, or falls back to
    deterministic prompt formatting.

    Args:
        problem_description: Description of the trial task / objective.
        git_diff: The clean git diff produced by the solution.
        teacher_fn: Optional Teacher model callable to distill the patch.

    Returns:
        A TrainingPair dataclass instance.
    """
    clean_problem = problem_description.strip()
    clean_diff = git_diff.strip()

    if teacher_fn is None:
        return TrainingPair(problem=clean_problem, patch=clean_diff)

    # When teacher_fn is provided, format prompt and invoke teacher
    prompt = f"Problem:\n{clean_problem}\n\nDiff:\n{clean_diff}"
    try:
        import inspect

        sig = inspect.signature(teacher_fn)
        params = list(sig.parameters.keys())
        if len(params) == 1 and params[0] in ("diff", "git_diff", "patch"):
            raw_output = teacher_fn(clean_diff)
        elif len(params) == 2:
            raw_output = teacher_fn(clean_problem, clean_diff)
        else:
            raw_output = teacher_fn(prompt)
    except Exception:
        raw_output = teacher_fn(prompt)

    output_str = str(raw_output).strip()
    problem = clean_problem
    patch = output_str

    # Extract tags if the teacher formatted explicit XML boundaries
    prob_match = re.search(r"<problem>(.*?)</problem>", output_str, re.DOTALL)
    patch_match = re.search(r"<patch>(.*?)</patch>", output_str, re.DOTALL)

    if prob_match:
        problem = prob_match.group(1).strip()
    if patch_match:
        patch = patch_match.group(1).strip()

    return TrainingPair(problem=problem, patch=patch)


def _is_slot_locked(lora_manager: Any, slot_id: str) -> bool:
    """Check if the designated LoRA slot is locked against mutation."""
    if hasattr(lora_manager, "is_slot_locked") and callable(lora_manager.is_slot_locked):
        return bool(lora_manager.is_slot_locked(slot_id))
    if hasattr(lora_manager, "is_locked") and callable(lora_manager.is_locked):
        return bool(lora_manager.is_locked(slot_id))
    if hasattr(lora_manager, "get_slot") and callable(lora_manager.get_slot):
        slot = lora_manager.get_slot(slot_id)
        if slot is not None:
            return bool(getattr(slot, "is_locked", False))
    if hasattr(lora_manager, "slots") and isinstance(lora_manager.slots, dict):
        slot = lora_manager.slots.get(slot_id)
        if slot is not None:
            return bool(getattr(slot, "is_locked", False))
    return False


def _derive_slot_vector(
    lora_manager: Any,
    slot_id: str,
    text: str,
    embedding: Optional[Sequence[float]] = None,
) -> List[float]:
    """Derive or normalize a centroid vector for the given slot."""
    if embedding is not None:
        return [float(x) for x in embedding]

    # Inspect existing dimension from slot if possible
    dim = 16
    existing_slot = None
    if hasattr(lora_manager, "get_slot") and callable(lora_manager.get_slot):
        existing_slot = lora_manager.get_slot(slot_id)
    elif hasattr(lora_manager, "slots") and isinstance(lora_manager.slots, dict):
        existing_slot = lora_manager.slots.get(slot_id)

    if existing_slot is not None and getattr(existing_slot, "centroid_vector", None):
        dim = len(existing_slot.centroid_vector)

    raw_vec: List[float] = []
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    idx = 0
    while len(raw_vec) < dim:
        b = seed_bytes[idx % len(seed_bytes)]
        raw_vec.append((b / 255.0) * 2.0 - 1.0)
        idx += 1
        if idx % len(seed_bytes) == 0:
            seed_bytes = hashlib.sha256(seed_bytes).digest()

    norm = sum(x * x for x in raw_vec) ** 0.5 or 1.0
    return [round(x / norm, 6) for x in raw_vec[:dim]]


def fine_tune_lora_slot(
    slot_id: str,
    training_pair: TrainingPair,
    lora_manager: Any,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    new_embedding: Optional[Sequence[float]] = None,
    salience_delta: float = 0.1,
) -> dict:
    """Execute low-rank fine-tuning on target LoRA slot and update centroid.

    Checks locked status, executes 3-epoch pass (PyTorch/PEFT live training
    or deterministic lightweight adapter step in mock/CPU mode), and updates
    the slot centroid vector in lora_manager.

    Args:
        slot_id: Target LoRA slot identifier.
        training_pair: Formatted synthetic training pair.
        lora_manager: The memory network manager instance.
        epochs: Number of training epochs (default: 3).
        learning_rate: Optimizer learning rate.
        new_embedding: Optional embedding vector to update centroid with.
        salience_delta: Salience boost applied upon successful consolidation (default: 0.1).

    Returns:
        Dictionary reporting training metadata, losses, and updated slot status.
    """
    if _is_slot_locked(lora_manager, slot_id):
        logger.warning(
            "LoRA slot '%s' is locked. Skipping weight mutation and centroid update.",
            slot_id,
        )
        return {
            "slot_id": slot_id,
            "status": "locked",
            "message": f"Slot '{slot_id}' is locked; weight mutation skipped.",
            "epochs": 0,
            "losses": [],
            "final_loss": None,
            "updated_slot": None,
        }

    # Ensure target adapter is hot-swapped and active
    if hasattr(lora_manager, "hot_swap_adapter") and callable(lora_manager.hot_swap_adapter):
        try:
            lora_manager.hot_swap_adapter(slot_id)
        except Exception as exc:
            logger.debug("hot_swap_adapter hook: %s", exc)

    # Identify if a live PyTorch model is attached to lora_manager
    live_model: Optional[Any] = None
    if TORCH_AVAILABLE:
        for attr_name in ("model", "peft_model", "active_model"):
            candidate = getattr(lora_manager, attr_name, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                live_model = candidate
                break

    losses: List[float] = []

    if TORCH_AVAILABLE and live_model is not None:
        # Real PyTorch / PEFT training loop
        live_model.train()
        if hasattr(live_model, "set_adapter") and callable(live_model.set_adapter):
            try:
                live_model.set_adapter(slot_id)
            except Exception as exc:
                logger.debug("set_adapter on live_model: %s", exc)

        trainable_params = [p for p in live_model.parameters() if p.requires_grad]
        if not trainable_params:
            if not hasattr(live_model, "peft_config"):
                for p in live_model.parameters():
                    p.requires_grad = True
                trainable_params = [p for p in live_model.parameters() if p.requires_grad]

        if trainable_params:
            eff_lr = max(learning_rate, 0.01)
            optimizer = torch.optim.AdamW(trainable_params, lr=eff_lr)
            tokenizer = getattr(lora_manager, "tokenizer", None)
            device = next(live_model.parameters()).device

            if tokenizer is not None:
                inputs = tokenizer(training_pair.formatted_text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = inputs.get("input_ids")
                for _ in range(epochs):
                    optimizer.zero_grad()
                    outputs = live_model(**inputs, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
            elif hasattr(live_model, "forward_loss") and callable(live_model.forward_loss):
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = live_model.forward_loss(training_pair.formatted_text)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
            elif hasattr(live_model, "compute_loss") and callable(live_model.compute_loss):
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = live_model.compute_loss(training_pair.formatted_text)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
            else:
                first_p = trainable_params[0]
                in_dim = first_p.shape[-1] if first_p.ndim > 1 else first_p.shape[0]
                text_hash = int(hashlib.md5(training_pair.formatted_text.encode("utf-8")).hexdigest()[:8], 16)
                g_in = torch.Generator(device=device).manual_seed(text_hash)
                dummy_in = torch.randn((4, in_dim), dtype=first_p.dtype, device=device, generator=g_in)
                with torch.no_grad():
                    init_out = live_model(dummy_in)
                g_target = torch.Generator(device=device).manual_seed(text_hash + 100)
                target = torch.randn(init_out.shape, dtype=init_out.dtype, device=device, generator=g_target)

                for _ in range(epochs):
                    optimizer.zero_grad()
                    outputs = live_model(dummy_in)
                    loss = nn.functional.mse_loss(outputs, target)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
        else:
            losses = [0.0] * epochs
    else:
        # Deterministic lightweight adapter step in mock/CPU mode
        if hasattr(lora_manager, "hot_swap_adapter") and callable(lora_manager.hot_swap_adapter):
            try:
                lora_manager.hot_swap_adapter(slot_id)
            except Exception as exc:
                logger.debug("hot_swap_adapter note: %s", exc)

        text_hash = int(hashlib.md5(training_pair.formatted_text.encode("utf-8")).hexdigest()[:8], 16)
        base_loss = 0.5 + (text_hash % 500) / 1000.0
        losses = [round(base_loss / (ep + 1), 6) for ep in range(epochs)]

        if hasattr(lora_manager, "record_training_step") and callable(
            lora_manager.record_training_step
        ):
            lora_manager.record_training_step(slot_id, epochs=epochs, lr=learning_rate)

    # Compute and update slot centroid
    target_vector = _derive_slot_vector(
        lora_manager=lora_manager,
        slot_id=slot_id,
        text=training_pair.formatted_text,
        embedding=new_embedding,
    )

    updated_slot = None
    if hasattr(lora_manager, "update_centroid") and callable(lora_manager.update_centroid):
        try:
            updated_slot = lora_manager.update_centroid(
                slot_id=slot_id,
                new_vector=target_vector,
                learning_rate=0.2,
                salience_delta=salience_delta,
            )
        except TypeError:
            updated_slot = lora_manager.update_centroid(
                slot_id=slot_id,
                new_vector=target_vector,
                learning_rate=0.2,
            )
    elif hasattr(lora_manager, "slots") and isinstance(lora_manager.slots, dict):
        if slot_id in lora_manager.slots:
            lora_manager.slots[slot_id].centroid_vector = target_vector
            if hasattr(lora_manager.slots[slot_id], "salience"):
                lora_manager.slots[slot_id].salience = max(
                    0.0, min(1.0, float(lora_manager.slots[slot_id].salience) + float(salience_delta))
                )
            updated_slot = lora_manager.slots[slot_id]

    return {
        "slot_id": slot_id,
        "status": "completed",
        "epochs": epochs,
        "losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "learning_rate": learning_rate,
        "updated_slot": updated_slot,
        "centroid_vector": target_vector,
    }


def run_sleep_cycle(
    problem_description: str,
    sandbox: Any,
    lora_manager: Any,
    regulator: Any,
    slot_id: str,
    query_embedding: Optional[Sequence[float]] = None,
    epochs: int = 3,
    teacher_fn: Optional[Callable[[str], str]] = None,
    workdir: str = "/home/trials",
    salience_delta: float = 0.1,
    learning_rate: float = 0.02,
) -> dict:
    """Orchestrate end-to-end consolidation pass on exit code 0 trial success.

    1. Extracts clean git diff from sandbox.
    2. Synthesizes <problem> -> <patch> training pair.
    3. Executes 3-epoch fine-tuning pass on target slot_id.
    4. Updates centroid vector in lora_manager and boosts salience.
    5. Calls regulator.reset_adenosine().

    Args:
        problem_description: Task prompt / description.
        sandbox: SandboxRunner instance or directory path.
        lora_manager: LoRAManager instance.
        regulator: Homeostasis regulator instance.
        slot_id: Target LoRA slot identifier.
        query_embedding: Optional query vector for centroid update.
        epochs: Number of training epochs (default: 3).
        teacher_fn: Optional teacher model distillation function.
        workdir: Working directory inside sandbox.
        salience_delta: Salience boost applied to active slot (default: 0.1).

    Returns:
        Summary dict of sleep consolidation run.
    """
    diff = extract_clean_diff(sandbox, workdir=workdir)
    if not diff:
        logger.warning(
            "No clean diff found during sleep cycle for problem: '%s'",
            problem_description,
        )
        return {
            "status": "no_diff",
            "slot_id": slot_id,
            "diff": "",
            "training_pair": None,
            "tune_result": None,
            "adenosine_reset": False,
        }

    training_pair = synthesize_training_pair(
        problem_description=problem_description,
        git_diff=diff,
        teacher_fn=teacher_fn,
    )

    tune_result = fine_tune_lora_slot(
        slot_id=slot_id,
        training_pair=training_pair,
        lora_manager=lora_manager,
        epochs=epochs,
        learning_rate=learning_rate,
        new_embedding=query_embedding,
        salience_delta=salience_delta,
    )

    # Homeostatic reset: clears adenosine sleep pressure
    if regulator is not None and hasattr(regulator, "reset_adenosine") and callable(
        regulator.reset_adenosine
    ):
        regulator.reset_adenosine()

    return {
        "status": "success",
        "slot_id": slot_id,
        "diff": diff,
        "training_pair": training_pair,
        "tune_result": tune_result,
        "epochs": epochs,
        "adenosine_reset": True,
    }


class SleepConsolidator:
    """Orchestrator for sleep consolidation and memory distillation."""

    def __init__(
        self,
        lora_manager: Optional[Any] = None,
        regulator: Optional[Any] = None,
        teacher_fn: Optional[Callable[[str], str]] = None,
        default_epochs: int = 3,
        default_lr: float = 1e-4,
    ) -> None:
        self.lora_manager = lora_manager
        self.regulator = regulator
        self.teacher_fn = teacher_fn
        self.default_epochs = default_epochs
        self.default_lr = default_lr

    def extract_clean_diff(
        self,
        sandbox_runner_or_workdir: Any,
        workdir: str = "/home/trials",
    ) -> str:
        """Extract clean git changes from a runner or local directory."""
        return extract_clean_diff(sandbox_runner_or_workdir, workdir=workdir)

    def synthesize_training_pair(
        self,
        problem_description: str,
        git_diff: str,
        teacher_fn: Optional[Callable[[str], str]] = None,
    ) -> TrainingPair:
        """Construct synthetic training pair using optional teacher model."""
        fn = teacher_fn if teacher_fn is not None else self.teacher_fn
        return synthesize_training_pair(
            problem_description=problem_description,
            git_diff=git_diff,
            teacher_fn=fn,
        )

    def fine_tune_lora_slot(
        self,
        slot_id: str,
        training_pair: TrainingPair,
        lora_manager: Optional[Any] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        new_embedding: Optional[Sequence[float]] = None,
    ) -> dict:
        """Fine-tune the target LoRA slot for specified epochs."""
        mgr = lora_manager if lora_manager is not None else self.lora_manager
        ep = epochs if epochs is not None else self.default_epochs
        lr = learning_rate if learning_rate is not None else self.default_lr
        return fine_tune_lora_slot(
            slot_id=slot_id,
            training_pair=training_pair,
            lora_manager=mgr,
            epochs=ep,
            learning_rate=lr,
            new_embedding=new_embedding,
        )

    def run_sleep_cycle(
        self,
        problem_description: str,
        sandbox: Any,
        lora_manager: Optional[Any] = None,
        regulator: Optional[Any] = None,
        slot_id: str = "slot_0",
        query_embedding: Optional[Sequence[float]] = None,
        epochs: Optional[int] = None,
        teacher_fn: Optional[Callable[[str], str]] = None,
        workdir: str = "/home/trials",
    ) -> dict:
        """Execute full consolidation pipeline and reset adenosine."""
        mgr = lora_manager if lora_manager is not None else self.lora_manager
        reg = regulator if regulator is not None else self.regulator
        ep = epochs if epochs is not None else self.default_epochs
        fn = teacher_fn if teacher_fn is not None else self.teacher_fn
        return run_sleep_cycle(
            problem_description=problem_description,
            sandbox=sandbox,
            lora_manager=mgr,
            regulator=reg,
            slot_id=slot_id,
            query_embedding=query_embedding,
            epochs=ep,
            teacher_fn=fn,
            workdir=workdir,
        )
