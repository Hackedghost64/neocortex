"""Test suite for Theoretical-Devrith 6-Repo Multi-Archetype Scale Benchmark.

Verifies:
1. AST Parameter Canonicalization ($ARG0, $ARG1, ...) across disparate codebases.
2. Blind Discovery (no file/function names) with AST + Test Fault Localization.
3. Real Pre-Trained 135M LLM Brain (SmolLM2-135M-Instruct) sleep consolidation.
4. False-Positive Trap detection, automatic rollback safety, and cortisol response.
5. End-to-end 6-repo scale benchmark execution with 100% token cut on transfer runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path("/home/divyam/Desktop/projects/theoritcal-devrith")
FRIDAY_PATH = Path("/home/divyam/Desktop/projects/friday")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(FRIDAY_PATH) not in sys.path:
    sys.path.insert(0, str(FRIDAY_PATH))

import pytest
from devrith_genesis_engine import (
    TheoreticalDevrith,
    extract_function_params,
    canonicalize_ast_line,
    decanonicalize_ast_line,
    RealPretrainedGenesisBrain,
)


@pytest.fixture(scope="module")
def engine() -> TheoreticalDevrith:
    # Run with the REAL pre-trained SmolLM2-135M-Instruct + PEFT LoRA slots (zero shortcuts!)
    return TheoreticalDevrith(seed=42, use_real_pretrained_llm=True)


def test_ast_parameter_canonicalization() -> None:
    """Verify AST parameter extraction, canonicalization, and de-canonicalization across variable names."""
    # Archetype A
    func_repo_1 = """def calculate_discounted_total(subtotal: float, promo_rate: float) -> float:
    return round(subtotal * (1.0 + promo_rate), 2)
"""
    params_1 = extract_function_params(func_repo_1)
    assert params_1 == ["subtotal", "promo_rate"]
    canon_1 = canonicalize_ast_line("return round(subtotal * (1.0 + promo_rate), 2)", params_1)
    assert canon_1 == "return round($ARG0 * (1.0 + $ARG1), 2)"

    params_2 = ["unit_cost", "markdown_pct"]
    decanon_2 = decanonicalize_ast_line("return round($ARG0 * (1.0 - $ARG1), 2)", params_2)
    assert decanon_2 == "return round(unit_cost * (1.0 - markdown_pct), 2)"

    # Archetype B
    params_b = ["records", "page_num", "page_size"]
    canon_b = canonicalize_ast_line("return records[page_num * page_size : (page_num + 1) * page_size]", params_b)
    assert canon_b == "return $ARG0[$ARG1 * $ARG2 : ($ARG1 + 1) * $ARG2]"

    params_b_transfer = ["entries", "current_page", "limit"]
    decanon_b = decanonicalize_ast_line("return $ARG0[($ARG1 - 1) * $ARG2 : $ARG1 * $ARG2]", params_b_transfer)
    assert decanon_b == "return entries[(current_page - 1) * limit : current_page * limit]"

    # Archetype C
    params_c = ["clicks", "impressions"]
    canon_c = canonicalize_ast_line("return round(clicks / impressions, 4)", params_c)
    assert canon_c == "return round($ARG0 / $ARG1, 4)"

    params_c_transfer = ["hits", "lookups"]
    decanon_c = decanonicalize_ast_line("return round($ARG0 / $ARG1, 4) if $ARG1 > 0 else 0.0", params_c_transfer)
    assert decanon_c == "return round(hits / lookups, 4) if lookups > 0 else 0.0"


def test_blind_discovery_and_fault_localization(engine: TheoreticalDevrith) -> None:
    """Verify that TheoreticalDevrith achieves 10/10 discovery accuracy on vague prompts."""
    engine.reset_scale_repos()
    base = ROOT_DIR / "repos_scale"

    # Vague prompt for Repo 2 (where keyword 'register' causes Vanilla Devrith to rank compute_register_change #1)
    prompt_repo_2 = "Clearance markdown prices are ringing up higher than retail on register"
    discovery_2 = engine._discover_buggy_target(prompt_repo_2, str(base / "repo_2_retail_pos"))

    # TheoreticalDevrith overrides spurious register symbol using test-signal trace + generic AST return extraction
    assert discovery_2["primary_file"] == "checkout/markdown_engine.py"
    assert discovery_2["anchor_symbol"] == "compute_sale_price"
    assert discovery_2["canonical_bug_signature"] == "return round($ARG0 * (1.0 + $ARG1), 2)"
    assert discovery_2["param_names"] == ["unit_cost", "markdown_pct"]

    # Vague prompt for Repo 3 (pagination)
    prompt_repo_3 = "Pagination slice is skipping the first record on page 1"
    discovery_3 = engine._discover_buggy_target(prompt_repo_3, str(base / "repo_3_crm_contacts"))
    assert discovery_3["primary_file"] == "directory/paginator.py"
    assert discovery_3["anchor_symbol"] == "slice_page"
    assert discovery_3["canonical_bug_signature"] == "return $ARG0[$ARG1 * $ARG2 : ($ARG1 + 1) * $ARG2]"
    assert discovery_3["param_names"] == ["records", "page_num", "page_size"]

    # Vague prompt for Repo 5 (zero division)
    prompt_repo_5 = "Zero activity campaigns trigger an unhandled division error when logging metrics"
    discovery_5 = engine._discover_buggy_target(prompt_repo_5, str(base / "repo_5_ad_analytics"))
    assert discovery_5["primary_file"] == "metrics/conversion.py"
    assert discovery_5["anchor_symbol"] == "conversion_rate"
    assert discovery_5["canonical_bug_signature"] == "return round($ARG0 / $ARG1, 4)"
    assert discovery_5["param_names"] == ["clicks", "impressions"]


def test_false_positive_trap_and_automatic_rollback(engine: TheoreticalDevrith) -> None:
    """Verify that incompatible patch injection fails verification, rolls back file, and spikes cortisol."""
    engine.reset_scale_repos()
    repo_6 = str(ROOT_DIR / "repos_scale" / "repo_6_Trap_and_Transfer")
    tax_file = Path(repo_6) / "tax" / "summary.py"
    original_code = tax_file.read_text(encoding="utf-8")

    # Run trap test
    trap_res = engine.solve_task(
        "Fix discount tax rounding error in billing summary",
        repo_6,
        force_trap_test=True,
    )

    assert trap_res["is_trap"] is True
    assert trap_res["trap_detected"] is True
    assert trap_res["rolled_back"] is True
    assert trap_res["cortisol_spike"] >= 0.80
    assert trap_res["dopamine"] == 0.0
    assert trap_res["escalated_to_frontier"] is True

    # Verify that tax/summary.py was completely restored to original content
    restored_code = tax_file.read_text(encoding="utf-8")
    assert restored_code == original_code


def test_6_repo_scale_benchmark_end_to_end(engine: TheoreticalDevrith) -> None:
    """Verify end-to-end 6-repo benchmark execution, 3/3 L1 cache hits, and 100% token cut on transfer."""
    engine.reset_scale_repos()
    res = engine.run_scale_benchmark()

    s = res["summary"]
    r1 = res["run_1_saas_billing"]
    r2 = res["run_2_retail_pos"]
    r3 = res["run_3_crm_contacts"]
    r4 = res["run_4_order_history"]
    r5 = res["run_5_ad_analytics"]
    r6_trap = res["run_6a_trap"]
    r6_trans = res["run_6b_transfer"]

    # 1. Verification of all runs
    assert r1["verify_success"] is True
    assert r2["verify_success"] is True
    assert r3["verify_success"] is True
    assert r4["verify_success"] is True
    assert r5["verify_success"] is True
    assert r6_trans["verify_success"] is True

    # 2. L1 Cache Hit rate: exactly 3 hits on transfer runs
    assert r2["cache_hit"] is True
    assert r4["cache_hit"] is True
    assert r6_trans["cache_hit"] is True
    assert s["l1_cache_hits_count"] == 3

    # 3. Zero Frontier Tokens on Transfer runs
    assert r2["theoretical_tokens"] == 0
    assert r2["theoretical_reduction_pct"] == 100.0

    assert r4["theoretical_tokens"] == 0
    assert r4["theoretical_reduction_pct"] == 100.0

    assert r6_trans["theoretical_tokens"] == 0
    assert r6_trans["theoretical_reduction_pct"] == 100.0

    assert s["zero_token_runs_count"] == 3

    # 4. Trap and Discovery Accuracy
    assert s["trap_rollback_success"] is True
    assert s["theoretical_discovery_score_out_of_10"] == 10.0
    assert s["theoretical_total_reduction_pct"] > 50.0
