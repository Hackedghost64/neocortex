"""Tests for Google Colaboratory Pro A100 MCP Bridge and Genesis Orchestration."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pytest

from engine.lora_manager import LoRAManager
from engine.remote_colab import ColabA100Bridge
from interfaces import CommandResult, TrainingPair
from orchestrator import GenesisOrchestrator, LocalSandboxRunner
from regulator.state import Homeostasis


class MockColabMCPServer:
    """Mock in-memory Colab MCP server simulating tool dispatches."""

    def __init__(self, runtimes: Optional[List[Dict[str, Any]]] = None) -> None:
        self.runtimes = runtimes if runtimes is not None else []
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.call_log: List[tuple[str, Dict[str, Any]]] = []
        self.should_unauthorize = False

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.call_log.append((tool_name, arguments))

        if self.should_unauthorize:
            return {
                "isError": True,
                "error": {
                    "code": 16,
                    "message": "Unauthorized: Google Colaboratory OAuth token expired or invalid.",
                },
            }

        if tool_name == "list_runtimes":
            return {"runtimes": list(self.runtimes)}

        elif tool_name == "create_runtime":
            spec = arguments.get("runtime", {}).get("runtimeSpec", {})
            accel = spec.get("accelerator", "A100")
            rt_name = f"runtimes/colab-{accel.lower()}-{len(self.runtimes) + 1}"
            rt_obj = {
                "name": rt_name,
                "runtimeSpec": spec,
                "connectionInfo": {
                    "token": "mock-token-xyz",
                    "url": "https://colab.research.google.com/runtime/mock",
                    "expireTime": "2026-11-08T00:00:00Z",
                },
            }
            self.runtimes.append(rt_obj)
            return {"runtime": rt_obj}

        elif tool_name == "create_session":
            parent = arguments["parent"]
            session_id = arguments.get("sessionId", "genesis-cortex")
            session_name = f"{parent}/sessions/{session_id}"
            sess_obj = {"name": session_name}
            self.sessions[session_name] = sess_obj
            return {"session": sess_obj}

        elif tool_name == "execute_code":
            session = arguments.get("session", "")
            code = arguments.get("code", "")

            # Remote initialization
            if "__GENESIS_INIT__" in code or "Genesis Remote A100 Initialization" in code:
                init_out = {
                    "status": "initialized",
                    "model_id": "Qwen/Qwen3.8-27B",
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "device": "cuda:0 (A100)",
                }
                return {
                    "stdout": "__GENESIS_INIT__" + json.dumps(init_out),
                    "session": session,
                }

            # Remote patch generation
            elif "__GENESIS_PATCH__" in code or "Genesis Patch Generation Step" in code:
                patch_str = (
                    "cat << 'EOF' > calc.py\n"
                    "def add(a, b):\n"
                    "    return a + b\n"
                    "EOF\n"
                    "python3 test_calc.py"
                )
                return {
                    "stdout": f"```bash\n{patch_str}\n```",
                    "patch_cmd": patch_str,
                    "session": session,
                }

            # Remote fine-tuning pass
            elif "__GENESIS_TUNE__" in code or "Genesis PEFT Fine-Tuning Pass" in code:
                tune_out = {
                    "status": "completed",
                    "slot_id": "slot_0",
                    "epochs": 3,
                    "losses": [0.6521, 0.3245, 0.1589],
                    "initial_loss": 0.6521,
                    "final_loss": 0.1589,
                    "device": "cuda:0 (A100)",
                }
                return {
                    "stdout": "__GENESIS_TUNE__" + json.dumps(tune_out),
                    "session": session,
                    "status": "completed",
                    "losses": [0.6521, 0.3245, 0.1589],
                    "initial_loss": 0.6521,
                    "final_loss": 0.1589,
                    "device": "cuda:0 (A100)",
                }

            return {"stdout": "", "session": session}

        elif tool_name == "delete_session":
            name = arguments.get("name", "")
            self.sessions.pop(name, None)
            return {"name": name}

        return {}


# ============================================================================
# 1. ColabA100Bridge Unit Tests
# ============================================================================


def test_bridge_reuses_existing_a100_runtime() -> None:
    """Verify bridge reuses an existing matching A100 runtime without calling create_runtime."""
    server = MockColabMCPServer(
        runtimes=[
            {
                "name": "runtimes/existing-colab-a100",
                "runtimeSpec": {
                    "variant": "VARIANT_GPU",
                    "accelerator": "A100",
                    "shape": "SHAPE_STANDARD",
                },
            }
        ]
    )
    bridge = ColabA100Bridge(mcp_caller=server)
    session_name = bridge.connect_or_create_runtime(
        accelerator="A100", session_id="test-session"
    )

    assert session_name == "runtimes/existing-colab-a100/sessions/test-session"
    assert bridge.runtime_name == "runtimes/existing-colab-a100"
    assert bridge.session_name == session_name

    called_tools = [tool for tool, _ in server.call_log]
    assert "list_runtimes" in called_tools
    assert "create_session" in called_tools
    assert "create_runtime" not in called_tools


def test_bridge_creates_a100_runtime_when_none_exist() -> None:
    """Verify bridge calls create_runtime with VARIANT_GPU and A100 when none exist."""
    server = MockColabMCPServer(runtimes=[])
    bridge = ColabA100Bridge(mcp_caller=server)
    session_name = bridge.connect_or_create_runtime(
        accelerator="A100", session_id="genesis-cortex"
    )

    assert "sessions/genesis-cortex" in session_name
    assert bridge.runtime_name == "runtimes/colab-a100-1"

    called_tools = [tool for tool, _ in server.call_log]
    assert "create_runtime" in called_tools
    create_args = [args for tool, args in server.call_log if tool == "create_runtime"][0]
    spec = create_args["runtime"]["runtimeSpec"]
    assert spec["variant"] == "VARIANT_GPU"
    assert spec["accelerator"] == "A100"
    assert spec["shape"] == "SHAPE_STANDARD"


def test_bridge_unauthorized_error_raises_permission_error() -> None:
    """Verify bridge raises clear PermissionError when Colab MCP returns Unauthorized."""
    server = MockColabMCPServer()
    server.should_unauthorize = True
    bridge = ColabA100Bridge(mcp_caller=server)

    with pytest.raises(PermissionError) as exc_info:
        bridge.connect_or_create_runtime()

    assert "Unauthorized" in str(exc_info.value)


def test_bridge_initialize_environment() -> None:
    """Verify bridge initializes environment with Qwen3.8-27B and 4-bit NF4 quantization."""
    server = MockColabMCPServer()
    bridge = ColabA100Bridge(mcp_caller=server)

    env_info = bridge.initialize_environment(
        model_id="Qwen/Qwen3.8-27B",
        load_in_4bit=True,
        lora_r=8,
        lora_alpha=16,
    )

    assert env_info["status"] == "initialized"
    assert env_info["model_id"] == "Qwen/Qwen3.8-27B"
    assert env_info["load_in_4bit"] is True
    assert env_info["device"] == "cuda:0 (A100)"
    assert env_info["lora_r"] == 8
    assert env_info["lora_alpha"] == 16


def test_bridge_generate_patch_command() -> None:
    """Verify bridge generates bash patch command via execute_code on Colab session."""
    server = MockColabMCPServer()
    bridge = ColabA100Bridge(mcp_caller=server)
    bridge.connect_or_create_runtime()

    patch_cmd = bridge.generate_patch_command(
        task_prompt="Fix arithmetic bug in calc.py",
        file_context={"calc.py": "def add(a, b):\n    return a + b + 1\n"},
        last_result=CommandResult(
            exit_code=1,
            stdout="",
            stderr="AssertionError: 6 != 5",
            command="python3 test_calc.py",
        ),
        slot_id="slot_1",
        test_cmd="python3 test_calc.py",
    )

    assert "calc.py" in patch_cmd
    assert "def add(a, b):" in patch_cmd
    assert "return a + b" in patch_cmd
    assert "python3 test_calc.py" in patch_cmd


def test_bridge_fine_tune_lora_slot_3_epochs() -> None:
    """Verify bridge sends training pair to Colab session and returns 3-epoch losses."""
    server = MockColabMCPServer()
    bridge = ColabA100Bridge(mcp_caller=server)
    bridge.connect_or_create_runtime()

    pair = TrainingPair(
        problem="Fix arithmetic bug in calc.py",
        patch="--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n-    return a + b + 1\n+    return a + b\n",
    )

    tune_result = bridge.fine_tune_lora_slot(
        slot_id="slot_0",
        training_pair=pair,
        epochs=3,
        learning_rate=1e-4,
    )

    assert tune_result["status"] == "completed"
    assert tune_result["slot_id"] == "slot_0"
    assert tune_result["epochs"] == 3
    assert len(tune_result["losses"]) == 3
    assert tune_result["losses"][0] > tune_result["losses"][-1]
    assert tune_result["device"] == "cuda:0 (A100)"


# ============================================================================
# 2. End-to-End Orchestrator + Sandbox + Colab Bridge Integration
# ============================================================================


def test_e2e_day1_orchestration_with_colab_bridge() -> None:
    """Verify Day 1 calc.py bug resolution and remote A100 sleep consolidation.

    Ensures:
    1. Wake cycle invokes bridge.generate_patch_command to fix calc.py.
    2. Sandbox executes patch and verifies test_calc.py exit code 0.
    3. Sleep cycle executes clean diff extraction, teacher distillation, and
       remote bridge.fine_tune_lora_slot on Colab A100 session.
    4. 3-epoch A100 loss curve is merged into sleep_report['tune_result'].
    5. Local SQLite slot centroid vector and salience are updated.
    6. Adenosine sleep pressure is reset to 0.0.
    """
    server = MockColabMCPServer()
    bridge = ColabA100Bridge(mcp_caller=server)
    sb = LocalSandboxRunner()
    sb.start()
    mgr = LoRAManager(use_mock=True)
    reg = Homeostasis()

    # Commander / Host Teacher model callable
    teacher_called = []

    def commander_teacher(problem: str, diff: str) -> str:
        teacher_called.append((problem, diff))
        return f"<problem>{problem}</problem>\n<patch>{diff}</patch>"

    orch = GenesisOrchestrator(
        sandbox=sb,
        regulator=reg,
        lora_manager=mgr,
        remote_bridge=bridge,
        teacher_fn=commander_teacher,
    )

    try:
        outcome = orch.run_trial(
            task_description="Fix arithmetic off-by-one bug in calc.py",
            test_cmd="python3 test_calc.py",
            similarity_threshold=0.75,
            decay_unvisited=True,
        )

        # 1. Trial Execution Outcome
        assert outcome.status == "consolidated"
        assert outcome.final_exit_code == 0
        assert outcome.steps_taken >= 1
        assert outcome.sleep_report is not None

        # 2. Remote Bridge Integration
        sleep_rep = outcome.sleep_report
        assert sleep_rep["status"] == "success"
        assert sleep_rep["adenosine_reset"] is True

        tune_res = sleep_rep["tune_result"]
        assert tune_res is not None
        assert tune_res["status"] == "completed"
        assert tune_res["device"] == "cuda:0 (A100)"
        assert len(tune_res["losses"]) == 3
        assert tune_res["initial_loss"] > tune_res["final_loss"]

        # 3. Teacher Model Distillation Verification
        assert len(teacher_called) == 1
        assert "calc.py" in teacher_called[0][1]

        # 4. Local SQLite Memory Network Verification
        slot_id = outcome.routed_slot_id
        assert slot_id is not None
        slot_rec = mgr.get_slot(slot_id)
        assert slot_rec is not None
        # Salience consolidated to 0.6 (0.5 initial + 0.1 delta)
        assert slot_rec.salience == pytest.approx(0.6, abs=1e-4)

        # 5. Homeostasis Adenosine Reset
        assert reg.adenosine == 0.0

    finally:
        sb.stop()
        mgr.close()
        bridge.delete_session()
