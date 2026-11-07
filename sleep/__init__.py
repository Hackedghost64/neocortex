"""Sleep and consolidation pipeline for Project Genesis."""

from sleep.curriculum import (
    CURRICULUM_SLOTS,
    CurriculumSlot,
    embed_text,
    run_teacher_bootcamp,
)
from sleep.distill import (
    SleepConsolidator,
    extract_clean_diff,
    fine_tune_lora_slot,
    run_sleep_cycle,
    synthesize_training_pair,
)

__all__ = [
    "CURRICULUM_SLOTS",
    "CurriculumSlot",
    "SleepConsolidator",
    "embed_text",
    "extract_clean_diff",
    "fine_tune_lora_slot",
    "run_sleep_cycle",
    "run_teacher_bootcamp",
    "synthesize_training_pair",
]
