"""Unified Execution Loop and Orchestrator for Project Genesis.

Wires the four verified modules:
1. Body (sandbox): Isolated workspace execution and command running.
2. Regulator: Homeostatic state tracking (dopamine, cortisol, adenosine).
3. Memory Network (engine): LoRA slot indexing, similarity routing, hot-swapping.
4. Sleep & Consolidation (sleep): Distillation of successful patches into LoRA weights.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from engine.lora_manager import LoRAManager
from interfaces import (
    CommandResult,
    HomeostasisProtocol,
    LoRAManagerProtocol,
    SandboxRunnerProtocol,
    SlotRecord,
    TrainingPair,
    TrialOutcome,
)
from regulator.state import Homeostasis
from sleep.distill import (
    extract_clean_diff,
    fine_tune_lora_slot,
    run_sleep_cycle,
    synthesize_training_pair,
)

logger = logging.getLogger(__name__)

# Standard broken scenario source definitions
TASK_DESCRIPTION: str = "Fix the arithmetic off-by-one bug in calc.py to make test_calc.py pass"

CALC_SOURCE: str = "def add(a, b):\n    return a + b + 1\n"

TEST_CALC_SOURCE: str = (
    "from calc import add\n\n\n"
    "def test_add():\n"
    "    assert add(2, 2) == 4\n\n\n"
    "if __name__ == '__main__':\n"
    "    test_add()\n"
    "    print('ALL TESTS PASSED')\n"
)


class LocalSandboxRunner(SandboxRunnerProtocol):
    """Lightweight local git workspace runner conforming to SandboxRunnerProtocol.

    Backed by tempfile.TemporaryDirectory and real subprocess execution for bash, git,
    and python3. Transparently maps container paths (e.g. '/home/trials') to the local
    workspace.
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> None:
        self.auto_cleanup = auto_cleanup
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        if workspace is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="genesis_sandbox_")
            self.workspace = os.path.abspath(self._temp_dir.name)
        else:
            self.workspace = os.path.abspath(workspace)
            os.makedirs(self.workspace, exist_ok=True)
        self.is_running: bool = False

    def start(self) -> str:
        """Mark the local sandbox as active and ensure workspace directory exists."""
        self.is_running = True
        os.makedirs(self.workspace, exist_ok=True)
        return self.workspace

    def _resolve_dir(self, directory: Optional[str]) -> str:
        """Translate /home/trials or arbitrary paths into local workspace filesystem path."""
        if not directory or directory in ("/home/trials", "/home/trials/", self.workspace):
            return self.workspace
        if os.path.isabs(directory):
            if directory.startswith("/home/trials"):
                rel = os.path.relpath(directory, "/home/trials")
                target = os.path.normpath(os.path.join(self.workspace, rel))
                os.makedirs(target, exist_ok=True)
                return target
            os.makedirs(directory, exist_ok=True)
            return directory
        target = os.path.normpath(os.path.join(self.workspace, directory))
        os.makedirs(target, exist_ok=True)
        return target

    def inject_files(
        self,
        files: Dict[str, Union[str, bytes]],
        dest_dir: str = "/home/trials",
    ) -> None:
        """Inject files into the resolved workspace directory."""
        if not self.is_running:
            raise RuntimeError("LocalSandboxRunner is not running. Call start() before injecting files.")
        target_dir = self._resolve_dir(dest_dir)
        for rel_path, content in files.items():
            full_path = os.path.join(target_dir, rel_path.lstrip("/"))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if isinstance(content, str):
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(full_path, "wb") as f:
                    f.write(content)

    def run_command(
        self,
        command: str,
        workdir: str = "/home/trials",
        timeout: int = 60,
    ) -> CommandResult:
        """Execute bash command in the resolved workspace directory via real subprocess."""
        if not self.is_running:
            raise RuntimeError("LocalSandboxRunner is not running. Call start() before executing commands.")
        cwd = self._resolve_dir(workdir)
        try:
            proc = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                command=command,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_str = (
                exc.stdout
                if isinstance(exc.stdout, str)
                else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
            )
            return CommandResult(
                command=command,
                stdout=stdout_str,
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=124,
            )
        except Exception as exc:
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(exc),
                exit_code=1,
            )

    def stop(self) -> None:
        """Stop runner and clean up temporary directory if auto_cleanup is True."""
        self.is_running = False
        if self._temp_dir is not None and self.auto_cleanup:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

    def __enter__(self) -> "LocalSandboxRunner":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def provision_trial(
    sandbox: SandboxRunnerProtocol,
    workdir: str = "/home/trials",
    task_text: Optional[str] = None,
    scenario_files: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Inject baseline broken scenario files and commit them into git inside the sandbox.

    Args:
        sandbox: An active or initialized sandbox conforming to SandboxRunnerProtocol.
        workdir: Target working directory inside sandbox.
        task_text: Optional custom task prompt.
        scenario_files: Optional custom dictionary of filename -> content.

    Returns:
        Dict mapping injected filenames to their string content.
    """
    if hasattr(sandbox, "is_running") and not sandbox.is_running:
        sandbox.start()

    task_content = task_text or TASK_DESCRIPTION
    if scenario_files is not None:
        files: Dict[str, str] = dict(scenario_files)
        if "task.md" not in files:
            files["task.md"] = task_content
    else:
        files = {
            "task.md": task_content,
            "calc.py": CALC_SOURCE,
            "test_calc.py": TEST_CALC_SOURCE,
        }
    sandbox.inject_files(files, dest_dir=workdir)

    # Initialize git repo, configure safe identity, and commit baseline
    file_list = " ".join(f"'{k}'" for k in files.keys())
    init_script = (
        "git config --global --add safe.directory '*' 2>/dev/null || true; "
        "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
        "  git init; "
        "fi; "
        "git config user.name 'Project Genesis'; "
        "git config user.email 'genesis@antigravity.ai'; "
        f"git add {file_list}; "
        "git commit -m 'Baseline broken scenario' 2>/dev/null || true"
    )
    res = sandbox.run_command(init_script, workdir=workdir)
    if res.exit_code != 0:
        logger.debug("Baseline git init note: %s (stderr: %s)", res.stdout, res.stderr)

    return files


def default_policy_fn(
    step: int,
    task: str,
    last_result: Optional[CommandResult] = None,
) -> str:
    """Default policy: run baseline test on step 0, then patch calc.py and re-test."""
    if step == 0:
        return "python3 test_calc.py"
    return (
        "python3 -c \"with open('calc.py', 'w') as f: f.write('def add(a, b):\\n    return a + b\\n')\" && "
        "python3 test_calc.py"
    )


class GenesisOrchestrator:
    """Unified execution orchestrator wiring sandbox, regulator, engine, and sleep modules."""

    def __init__(
        self,
        sandbox: Optional[SandboxRunnerProtocol] = None,
        regulator: Optional[HomeostasisProtocol] = None,
        lora_manager: Optional[LoRAManagerProtocol] = None,
        workdir: str = "/home/trials",
        teacher_fn: Optional[Callable[[str], str]] = None,
        remote_bridge: Optional[Any] = None,
        sleep_epochs: int = 3,
        sleep_lr: float = 0.02,
    ) -> None:
        self.sandbox = sandbox
        self.regulator = regulator if regulator is not None else Homeostasis()
        self.lora_manager = lora_manager if lora_manager is not None else LoRAManager()
        self.workdir = workdir
        self.teacher_fn = teacher_fn
        self.remote_bridge = remote_bridge
        self.sleep_epochs = sleep_epochs
        self.sleep_lr = sleep_lr

    def _ensure_sandbox(self, target_workdir: str) -> SandboxRunnerProtocol:
        """Ensure active sandbox runner instance is available."""
        if self.sandbox is None:
            self.sandbox = LocalSandboxRunner(workspace=target_workdir)
        if hasattr(self.sandbox, "is_running") and not self.sandbox.is_running:
            self.sandbox.start()
        return self.sandbox

    def provision_trial(
        self,
        workdir: Optional[str] = None,
        task_text: Optional[str] = None,
        scenario_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Provision the baseline broken scenario into the sandbox workspace."""
        target_workdir = workdir or self.workdir
        sb = self._ensure_sandbox(target_workdir)
        return provision_trial(
            sb,
            workdir=target_workdir,
            task_text=task_text,
            scenario_files=scenario_files,
        )

    def run_trial(
        self,
        query_embedding: Optional[Sequence[float]] = None,
        max_steps: int = 10,
        policy_fn: Optional[Callable[[int, str, Optional[CommandResult]], str]] = None,
        task_description: Optional[str] = None,
        workdir: Optional[str] = None,
        auto_provision: bool = True,
        scenario_files: Optional[Dict[str, str]] = None,
        test_cmd: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        decay_unvisited: bool = True,
        remote_bridge: Optional[Any] = None,
        teacher_fn: Optional[Callable[[str], str]] = None,
        sleep_epochs: Optional[int] = None,
        sleep_lr: Optional[float] = None,
    ) -> TrialOutcome:
        """Run the end-to-end Genesis execution loop (wake cycle, evaluation, and sleep trigger).

        Args:
            query_embedding: Optional vector to route to the appropriate LoRA slot.
            max_steps: Maximum action iterations before termination.
            policy_fn: Policy callable producing bash command strings given (step, task, last_res).
            task_description: Task prompt / description.
            workdir: Working directory inside sandbox.
            auto_provision: Whether to inject baseline broken files before starting the loop.
            scenario_files: Optional custom scenario files mapping filename -> content.
            test_cmd: Optional initial verification test command.
            similarity_threshold: Optional cosine similarity cutoff for dynamic slot allocation.
            decay_unvisited: Whether to decay unvisited slot saliences after the cycle.
            remote_bridge: Optional ColabA100Bridge instance for remote inference & fine-tuning.
            teacher_fn: Optional Teacher model callable to distill problem -> patch training pairs.

        Returns:
            Structured TrialOutcome with execution statistics and consolidation status.
        """
        active_bridge = remote_bridge if remote_bridge is not None else self.remote_bridge
        active_teacher = teacher_fn if teacher_fn is not None else self.teacher_fn
        target_workdir = workdir or self.workdir
        sandbox = self._ensure_sandbox(target_workdir)
        task_prompt = task_description or TASK_DESCRIPTION

        # 1. Provision trial files and commit baseline if requested
        if auto_provision:
            provision_trial(
                sandbox,
                workdir=target_workdir,
                task_text=task_prompt,
                scenario_files=scenario_files,
            )

        # 2. Derive or prepare query embedding and route to LoRA slot
        if query_embedding is None:
            # Match existing slot dimension or default to 16
            slots = getattr(self.lora_manager, "list_slots", lambda: [])()
            dim = len(slots[0].centroid_vector) if slots else 16
            query_embedding = [0.1] * dim

        if similarity_threshold is not None and hasattr(self.lora_manager, "route_or_allocate"):
            routed_slot, similarity_score, _ = self.lora_manager.route_or_allocate(
                query_vector=query_embedding,
                similarity_threshold=similarity_threshold,
            )
        else:
            routed_slot, similarity_score = self.lora_manager.route_by_similarity(query_embedding)
            if routed_slot is None:
                # If memory network has no slots registered yet, initialize a base slot
                routed_slot = self.lora_manager.init_slot(
                    slot_id="slot_0",
                    centroid_vector=list(query_embedding),
                    salience=0.5,
                )
                similarity_score = 1.0

        slot_id = routed_slot.slot_id
        self.lora_manager.hot_swap_adapter(slot_id)

        def _maybe_decay() -> None:
            if decay_unvisited and hasattr(self.lora_manager, "decay_unvisited_salience") and callable(
                self.lora_manager.decay_unvisited_salience
            ):
                self.lora_manager.decay_unvisited_salience(active_slot_id=slot_id)

        # 3. The Wake Cycle (Iterative Attempt Loop)
        last_result: Optional[CommandResult] = None
        last_error_code: Optional[int] = None
        steps_taken: int = 0
        final_exit_code: int = 1

        for step in range(max_steps):
            steps_taken += 1

            # Generate action command from policy
            if policy_fn is not None:
                try:
                    cmd = policy_fn(step, task_prompt, last_result)
                except TypeError:
                    try:
                        cmd = policy_fn(step, task_prompt)  # type: ignore[call-arg]
                    except TypeError:
                        cmd = policy_fn(step)  # type: ignore[call-arg]
            elif active_bridge is not None:
                file_ctx = scenario_files if scenario_files is not None else {
                    "calc.py": CALC_SOURCE,
                    "test_calc.py": TEST_CALC_SOURCE,
                }
                cmd = active_bridge.generate_patch_command(
                    task_prompt=task_prompt,
                    file_context=file_ctx,
                    last_result=last_result,
                    slot_id=slot_id,
                    test_cmd=test_cmd or "python3 test_calc.py",
                )
            else:
                if test_cmd is not None and step == 0:
                    cmd = test_cmd
                else:
                    cmd = default_policy_fn(step, task_prompt, last_result)

            # Execute command in sandbox
            cmd_result = sandbox.run_command(cmd, workdir=target_workdir)
            last_result = cmd_result
            final_exit_code = cmd_result.exit_code

            # Command tick: increment adenosine
            if hasattr(self.regulator, "record_tool_call") and callable(self.regulator.record_tool_call):
                self.regulator.record_tool_call()
            elif hasattr(self.regulator, "adenosine"):
                self.regulator.adenosine = min(1.0, float(self.regulator.adenosine) + 0.1)

            # 4. Evaluation & The Sleep Trigger
            if cmd_result.exit_code == 0:
                # Succeeded: Boost dopamine by +0.4
                current_dopamine = float(getattr(self.regulator, "dopamine", 0.0))
                self.regulator.dopamine = min(1.0, current_dopamine + 0.4)

                # Check if slot should be locked (dopamine >= 0.8)
                should_lock = (
                    self.regulator.should_lock_slot()
                    if hasattr(self.regulator, "should_lock_slot") and callable(self.regulator.should_lock_slot)
                    else float(getattr(self.regulator, "dopamine", 0.0)) >= 0.8
                )

                # Execute sleep consolidation cycle first before locking the slot
                active_epochs = sleep_epochs if sleep_epochs is not None else self.sleep_epochs
                active_lr = sleep_lr if sleep_lr is not None else self.sleep_lr
                sleep_report = run_sleep_cycle(
                    problem_description=task_prompt,
                    sandbox=sandbox,
                    lora_manager=self.lora_manager,
                    regulator=self.regulator,
                    slot_id=slot_id,
                    query_embedding=query_embedding,
                    epochs=active_epochs,
                    learning_rate=active_lr,
                    teacher_fn=active_teacher,
                    workdir=target_workdir,
                )

                if active_bridge is not None:
                    # Invoke remote fine-tuning if slot is not locked
                    is_locked = False
                    if hasattr(self.lora_manager, "is_slot_locked") and callable(self.lora_manager.is_slot_locked):
                        is_locked = bool(self.lora_manager.is_slot_locked(slot_id))
                    elif hasattr(self.lora_manager, "is_locked") and callable(self.lora_manager.is_locked):
                        is_locked = bool(self.lora_manager.is_locked(slot_id))

                    if not is_locked and sleep_report.get("training_pair") is not None:
                        remote_tune = active_bridge.fine_tune_lora_slot(
                            slot_id=slot_id,
                            training_pair=sleep_report["training_pair"],
                            epochs=3,
                        )
                        if isinstance(sleep_report.get("tune_result"), dict):
                            sleep_report["tune_result"].update(remote_tune)
                        else:
                            sleep_report["tune_result"] = remote_tune

                # Lock slot after consolidation if dopamine >= 0.8 threshold reached
                slot_locked = False
                if should_lock:
                    if hasattr(self.lora_manager, "lock_slot") and callable(self.lora_manager.lock_slot):
                        self.lora_manager.lock_slot(slot_id)
                        slot_locked = True

                _maybe_decay()

                return TrialOutcome(
                    status="consolidated",
                    routed_slot_id=slot_id,
                    similarity_score=similarity_score,
                    steps_taken=steps_taken,
                    final_exit_code=cmd_result.exit_code,
                    dopamine=float(getattr(self.regulator, "dopamine", 0.0)),
                    cortisol=float(getattr(self.regulator, "cortisol", 0.0)),
                    adenosine=float(getattr(self.regulator, "adenosine", 0.0)),
                    slot_locked=slot_locked,
                    sleep_report=sleep_report,
                )

            else:
                # Failed execution (exit_code != 0)
                is_repeat = (last_error_code is not None and last_error_code == cmd_result.exit_code)
                last_error_code = cmd_result.exit_code

                if hasattr(self.regulator, "on_command_result") and callable(self.regulator.on_command_result):
                    try:
                        self.regulator.on_command_result(
                            cmd_result.exit_code,
                            is_repeat_error=is_repeat,
                            accumulate_adenosine=False,
                        )
                    except TypeError:
                        self.regulator.on_command_result(cmd_result.exit_code, is_repeat_error=is_repeat)
                else:
                    boost = 0.45 if is_repeat else 0.25
                    self.regulator.cortisol = min(1.0, float(getattr(self.regulator, "cortisol", 0.0)) + boost)
                    if hasattr(self.regulator, "dopamine"):
                        self.regulator.dopamine = max(0.0, float(self.regulator.dopamine) - 0.05)

                # Check cortisol termination threshold (>= 0.8)
                if (
                    hasattr(self.regulator, "should_terminate_trial")
                    and callable(self.regulator.should_terminate_trial)
                    and self.regulator.should_terminate_trial()
                ):
                    _maybe_decay()
                    return TrialOutcome(
                        status="terminated_cortisol",
                        routed_slot_id=slot_id,
                        similarity_score=similarity_score,
                        steps_taken=steps_taken,
                        final_exit_code=cmd_result.exit_code,
                        dopamine=float(getattr(self.regulator, "dopamine", 0.0)),
                        cortisol=float(getattr(self.regulator, "cortisol", 0.0)),
                        adenosine=float(getattr(self.regulator, "adenosine", 0.0)),
                        slot_locked=False,
                        sleep_report=None,
                    )

                # Check adenosine sleep pressure trigger (>= 1.0)
                if (
                    hasattr(self.regulator, "should_trigger_sleep")
                    and callable(self.regulator.should_trigger_sleep)
                    and self.regulator.should_trigger_sleep()
                ):
                    _maybe_decay()
                    return TrialOutcome(
                        status="sleep_triggered_adenosine",
                        routed_slot_id=slot_id,
                        similarity_score=similarity_score,
                        steps_taken=steps_taken,
                        final_exit_code=cmd_result.exit_code,
                        dopamine=float(getattr(self.regulator, "dopamine", 0.0)),
                        cortisol=float(getattr(self.regulator, "cortisol", 0.0)),
                        adenosine=float(getattr(self.regulator, "adenosine", 0.0)),
                        slot_locked=False,
                        sleep_report=None,
                    )

        # Maximum trial step limit reached without success or early termination
        _maybe_decay()
        return TrialOutcome(
            status="max_steps_exceeded",
            routed_slot_id=slot_id,
            similarity_score=similarity_score,
            steps_taken=steps_taken,
            final_exit_code=final_exit_code,
            dopamine=float(getattr(self.regulator, "dopamine", 0.0)),
            cortisol=float(getattr(self.regulator, "cortisol", 0.0)),
            adenosine=float(getattr(self.regulator, "adenosine", 0.0)),
            slot_locked=False,
            sleep_report=None,
        )


def run_trial(
    sandbox: Optional[SandboxRunnerProtocol] = None,
    regulator: Optional[HomeostasisProtocol] = None,
    lora_manager: Optional[LoRAManagerProtocol] = None,
    query_embedding: Optional[Sequence[float]] = None,
    max_steps: int = 10,
    policy_fn: Optional[Callable[[int, str, Optional[CommandResult]], str]] = None,
    task_description: Optional[str] = None,
    workdir: str = "/home/trials",
    teacher_fn: Optional[Callable[[str], str]] = None,
    auto_provision: bool = True,
    remote_bridge: Optional[Any] = None,
    sleep_epochs: Optional[int] = None,
    sleep_lr: Optional[float] = None,
) -> TrialOutcome:
    """Convenience function running an end-to-end trial through GenesisOrchestrator."""
    orchestrator = GenesisOrchestrator(
        sandbox=sandbox,
        regulator=regulator,
        lora_manager=lora_manager,
        workdir=workdir,
        teacher_fn=teacher_fn,
        remote_bridge=remote_bridge,
        sleep_epochs=sleep_epochs or 3,
        sleep_lr=sleep_lr or 0.02,
    )
    return orchestrator.run_trial(
        query_embedding=query_embedding,
        max_steps=max_steps,
        policy_fn=policy_fn,
        task_description=task_description,
        workdir=workdir,
        auto_provision=auto_provision,
        remote_bridge=remote_bridge,
        sleep_epochs=sleep_epochs,
        sleep_lr=sleep_lr,
    )


__all__ = [
    "CALC_SOURCE",
    "GenesisOrchestrator",
    "LocalSandboxRunner",
    "TASK_DESCRIPTION",
    "TEST_CALC_SOURCE",
    "TrialOutcome",
    "default_policy_fn",
    "provision_trial",
    "run_trial",
]
