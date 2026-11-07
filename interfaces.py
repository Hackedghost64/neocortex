"""Strict architectural interfaces and shared contracts for Project Genesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@dataclass(frozen=True)
class CommandResult:
    """Result of executing a command inside the Docker execution sandbox."""

    command: str
    stdout: str
    stderr: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        """Return True when the command exited with code 0."""
        return self.exit_code == 0

    @property
    def binary_exit_code(self) -> int:
        """Normalized binary exit status: 0 for success, 1 for any non-zero failure."""
        return 0 if self.exit_code == 0 else 1


@dataclass
class SlotRecord:
    """Metadata row stored in the SQLite LoRA index."""

    slot_id: str
    centroid_vector: List[float]
    salience: float = 0.0
    is_locked: bool = False
    adapter_path: Optional[str] = None


@dataclass(frozen=True)
class TrainingPair:
    """Synthetic <problem> -> <patch> pair produced during sleep consolidation."""

    problem: str
    patch: str
    formatted_text: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formatted_text",
            f"<problem>\n{self.problem.strip()}\n</problem>\n<patch>\n{self.patch.strip()}\n</patch>",
        )


@runtime_checkable
class SandboxRunnerProtocol(Protocol):
    """Contract for the Body (Docker Execution Sandbox)."""

    def start(self) -> str: ...
    def inject_files(self, files: Dict[str, str | bytes], dest_dir: str = "/home/trials") -> None: ...
    def run_command(self, command: str, workdir: str = "/home/trials", timeout: int = 60) -> CommandResult: ...
    def stop(self) -> None: ...


@runtime_checkable
class HomeostasisProtocol(Protocol):
    """Contract for the Homeostatic Regulator."""

    dopamine: float
    cortisol: float
    adenosine: float

    def on_command_result(self, exit_code: int, is_repeat_error: bool = False) -> None: ...
    def step_decay(self) -> None: ...
    def should_lock_slot(self) -> bool: ...
    def should_terminate_trial(self) -> bool: ...
    def should_trigger_sleep(self) -> bool: ...
    def reset_adenosine(self) -> None: ...


@runtime_checkable
class LoRAManagerProtocol(Protocol):
    """Contract for the Memory Network LoRA slot manager."""

    def init_slot(
        self,
        slot_id: str,
        centroid_vector: Sequence[float],
        salience: float = 0.5,
        is_locked: bool = False,
    ) -> SlotRecord: ...
    def route_by_similarity(self, query_vector: Sequence[float]) -> Tuple[Optional[SlotRecord], float]: ...
    def hot_swap_adapter(self, slot_id: str) -> str: ...
    def lock_slot(self, slot_id: str) -> None: ...
    def save_adapter(self, slot_id: str, save_directory: Optional[str] = None) -> str: ...
    def update_centroid(
        self,
        slot_id: str,
        new_vector: Sequence[float],
        learning_rate: float = 0.2,
        salience_delta: float = 0.0,
    ) -> SlotRecord: ...
    def decay_unvisited_salience(
        self,
        active_slot_id: Optional[str] = None,
        decay_step: float = 0.08,
    ) -> Dict[str, float]: ...
    def route_or_allocate(
        self,
        query_vector: Sequence[float],
        similarity_threshold: float = 0.75,
        slot_prefix: str = "slot",
    ) -> Tuple[SlotRecord, float, bool]: ...


@dataclass
class TrialOutcome:
    """Structured outcome returned at the conclusion of a Genesis orchestrator trial."""

    status: str
    routed_slot_id: Optional[str]
    similarity_score: float
    steps_taken: int
    final_exit_code: int
    dopamine: float
    cortisol: float
    adenosine: float
    slot_locked: bool
    sleep_report: Optional[Dict[str, Any]] = None

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

