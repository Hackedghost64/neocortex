"""Test suite for Theoretical-Devrith: Devrith + Genesis L1 LoRA Cache 2-Repo Benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Add paths
THEORETICAL_DEVRITH_DIR = Path("/home/divyam/Desktop/projects/theoritcal-devrith")
FRIDAY_PATH = Path("/home/divyam/Desktop/projects/friday")
if str(THEORETICAL_DEVRITH_DIR) not in sys.path:
    sys.path.insert(0, str(THEORETICAL_DEVRITH_DIR))
if str(FRIDAY_PATH) not in sys.path:
    sys.path.insert(0, str(FRIDAY_PATH))

import pytest
from devrith_genesis_engine import DevrithGenesisEngine


@pytest.fixture(scope="module")
def engine() -> DevrithGenesisEngine:
    return DevrithGenesisEngine(seed=42)


def test_baseline_repo_bug_states(engine: DevrithGenesisEngine) -> None:
    """Verify that both repositories start with the expected failing bug state."""
    engine.reset_repos()
    repo_alpha = str(THEORETICAL_DEVRITH_DIR / "repos" / "repo_alpha_payments")
    repo_beta = str(THEORETICAL_DEVRITH_DIR / "repos" / "repo_beta_inventory")

    # Repo Alpha test must fail initially
    proc_alpha = subprocess.run(
        ["python3", "-m", "pytest", "-q", "tests/test_billing.py"],
        cwd=repo_alpha,
        capture_output=True,
        text=True,
    )
    assert proc_alpha.returncode != 0
    assert "assert 230.0 == 170.0" in proc_alpha.stdout or "FAILED" in proc_alpha.stdout

    # Repo Beta test must fail initially
    proc_beta = subprocess.run(
        ["python3", "-m", "pytest", "-q", "tests/test_pricing.py"],
        cwd=repo_beta,
        capture_output=True,
        text=True,
    )
    assert proc_beta.returncode != 0
    assert "assert 600.0 == 400.0" in proc_beta.stdout or "FAILED" in proc_beta.stdout


def test_2_repo_devrith_genesis_token_reduction_benchmark(engine: DevrithGenesisEngine) -> None:
    """Verify full 2-repo benchmark execution, LoRA loss drop, dopamine spike, and 100% token cut on Run 2."""
    engine.reset_repos()
    res = engine.run_benchmark()

    r1 = res["run_1_alpha"]
    r2 = res["run_2_beta"]
    summ = res["summary"]

    # 1. Run 1: L1 Cache Miss & Sleep Consolidation
    assert r1["cache_hit"] is False
    assert r1["lora_loss_start"] > 4.0
    assert r1["lora_loss_end"] < 0.10
    assert r1["dopamine_level"] >= 0.80
    assert r1["slot_locked"] is True
    assert r1["verify_success"] is True

    # 2. Run 2: L1 Cache Hit & Zero-Token Parametric Generation
    assert r2["cache_hit"] is True
    assert r2["similarity"] >= 0.98
    assert r2["theoretical_tokens"] == 0
    assert r2["theoretical_reduction_pct"] == 100.0
    assert "return round(amount * (1.0 - discount_rate), 2)" in r2["generated_patch"]
    assert r2["verify_success"] is True

    # 3. Overall Benchmark Token Savings
    assert summ["run_2_frontier_tokens"] == 0
    assert summ["run_2_reduction_pct"] == 100.0
    assert summ["theoretical_total_reduction_pct"] > 70.0
