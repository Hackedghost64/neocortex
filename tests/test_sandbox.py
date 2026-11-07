"""Unit tests for DockerSandboxRunner in Project Genesis."""

from __future__ import annotations

import io
import subprocess
import tarfile
from unittest.mock import MagicMock, call, patch

import pytest

from interfaces import CommandResult, SandboxRunnerProtocol
from sandbox.docker_runner import DockerSandboxRunner


# ============================================================================
# 1. Protocol & Conformance
# ============================================================================


def test_protocol_conformance():
    """Verify DockerSandboxRunner conforms to SandboxRunnerProtocol."""
    runner = DockerSandboxRunner(backend="subprocess")
    assert isinstance(runner, SandboxRunnerProtocol)


# ============================================================================
# 2. SDK Backend Unit Tests
# ============================================================================


class TestDockerSandboxRunnerSDK:
    """Test DockerSandboxRunner using mocked Docker SDK client."""

    @pytest.fixture
    def mock_container(self):
        container = MagicMock()
        container.id = "mock_container_id_12345"

        # exec_run returns ExecResult(exit_code, output)
        exec_res = MagicMock()
        exec_res.exit_code = 0
        exec_res.output = (b"mock_stdout\n", b"")
        container.exec_run.return_value = exec_res
        return container

    @pytest.fixture
    def mock_client(self, mock_container):
        client = MagicMock()
        client.containers.run.return_value = mock_container
        return client

    def test_startup_and_workspace_init(self, mock_client, mock_container):
        runner = DockerSandboxRunner(
            image="genesis:latest",
            workspace="/home/trials",
            container_name="test_box",
            client=mock_client,
        )

        assert not runner.is_running
        assert runner.container_id is None

        cid = runner.start()
        assert cid == "mock_container_id_12345"
        assert runner.is_running
        assert runner.container_id == "mock_container_id_12345"

        mock_client.containers.run.assert_called_once_with(
            image="genesis:latest",
            command=["sleep", "infinity"],
            detach=True,
            working_dir="/home/trials",
            name="test_box",
            network_disabled=False,
        )

        # Workspace init command must have been executed
        assert mock_container.exec_run.call_count >= 1
        first_call = mock_container.exec_run.call_args_list[0]
        cmd_arg = first_call.kwargs.get("cmd", first_call.args[0] if first_call.args else None)
        assert "/bin/bash" in cmd_arg
        assert "mkdir -p /home/trials" in cmd_arg[2]
        assert "git init" in cmd_arg[2]

    def test_start_is_idempotent(self, mock_client):
        runner = DockerSandboxRunner(client=mock_client)
        cid1 = runner.start()
        cid2 = runner.start()
        assert cid1 == cid2
        # containers.run should only be called once
        assert mock_client.containers.run.call_count == 1

    def test_inject_files(self, mock_client, mock_container):
        runner = DockerSandboxRunner(client=mock_client)
        runner.start()

        files = {
            "main.py": "print('hello')",
            "nested/module.py": b"x = 42\n",
        }
        runner.inject_files(files, dest_dir="/home/trials")

        mock_container.put_archive.assert_called_once()
        dest_dir, tar_bytes = mock_container.put_archive.call_args.args
        assert dest_dir == "/home/trials"

        # Verify tar content integrity
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            names = tar.getnames()
            assert "main.py" in names
            assert "nested" in names
            assert "nested/module.py" in names

            main_file = tar.extractfile("main.py")
            assert main_file is not None
            assert main_file.read() == b"print('hello')"

            nested_file = tar.extractfile("nested/module.py")
            assert nested_file is not None
            assert nested_file.read() == b"x = 42\n"

    def test_run_command_success(self, mock_client, mock_container):
        mock_res = MagicMock()
        mock_res.exit_code = 0
        mock_res.output = (b"output text\n", b"")
        mock_container.exec_run.return_value = mock_res

        runner = DockerSandboxRunner(client=mock_client)
        runner.start()

        result = runner.run_command("echo test", workdir="/home/trials/sub")
        assert isinstance(result, CommandResult)
        assert result.command == "echo test"
        assert result.stdout == "output text\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.succeeded is True
        assert result.binary_exit_code == 0

    def test_run_command_failure_and_binary_exit_code(self, mock_client, mock_container):
        mock_res = MagicMock()
        mock_res.exit_code = 42
        mock_res.output = (b"partial stdout", b"some error details\n")
        mock_container.exec_run.return_value = mock_res

        runner = DockerSandboxRunner(client=mock_client)
        runner.start()

        result = runner.run_command("bad_cmd")
        assert result.exit_code == 42
        assert result.succeeded is False
        assert result.binary_exit_code == 1
        assert result.stdout == "partial stdout"
        assert result.stderr == "some error details\n"

    def test_run_command_timeout(self, mock_client, mock_container):
        import time

        def slow_exec(*args, **kwargs):
            time.sleep(0.5)
            return MagicMock(exit_code=0, output=(b"", b""))

        mock_container.exec_run.side_effect = slow_exec

        runner = DockerSandboxRunner(client=mock_client)
        # Avoid hanging on start workspace init:
        with patch.object(runner, "_ensure_workspace_initialized"):
            runner.start()

        result = runner.run_command("sleep 10", timeout=1)  # small timeout
        # Using timeout=0 or near-0 triggers concurrent.futures.TimeoutError
        assert isinstance(result, CommandResult)

    def test_stop_teardown(self, mock_client, mock_container):
        runner = DockerSandboxRunner(client=mock_client)
        runner.start()
        assert runner.is_running

        runner.stop()
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)
        assert not runner.is_running
        assert runner.container_id is None

        # Multiple stops should not raise
        runner.stop()

    def test_context_manager(self, mock_client, mock_container):
        runner = DockerSandboxRunner(client=mock_client)
        with runner as r:
            assert r.is_running
            assert r.container_id == "mock_container_id_12345"

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert not runner.is_running


# ============================================================================
# 3. Subprocess Backend Unit Tests
# ============================================================================


class TestDockerSandboxRunnerSubprocess:
    """Test DockerSandboxRunner using mocked subprocess CLI calls."""

    @pytest.fixture
    def mock_subprocess_run(self):
        with patch("subprocess.run") as mock_run:
            def side_effect(cmd, *args, **kwargs):
                cmd_list = list(cmd)
                # docker run -d ...
                if len(cmd_list) >= 2 and cmd_list[0] == "docker" and cmd_list[1] == "run":
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout="mock_cli_container_abcdef\n",
                        stderr="",
                    )
                # docker exec ...
                elif len(cmd_list) >= 2 and cmd_list[0] == "docker" and cmd_list[1] == "exec":
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout="cli command output\n",
                        stderr="",
                    )
                # docker cp ...
                elif len(cmd_list) >= 2 and cmd_list[0] == "docker" and cmd_list[1] == "cp":
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout="",
                        stderr="",
                    )
                # docker rm ...
                elif len(cmd_list) >= 2 and cmd_list[0] == "docker" and cmd_list[1] == "rm":
                    return subprocess.CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout="removed\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

            mock_run.side_effect = side_effect
            yield mock_run

    def test_subprocess_startup_and_git_init(self, mock_subprocess_run):
        runner = DockerSandboxRunner(
            image="python:3.11-slim",
            workspace="/home/trials",
            network_disabled=True,
            backend="subprocess",
        )

        cid = runner.start()
        assert cid == "mock_cli_container_abcdef"
        assert runner.is_running
        assert runner.backend == "subprocess"

        # Verify docker run flags
        first_call = mock_subprocess_run.call_args_list[0]
        run_cmd = first_call.args[0]
        assert run_cmd[:3] == ["docker", "run", "-d"]
        assert "--network" in run_cmd
        assert "none" in run_cmd
        assert "--workdir" in run_cmd
        assert "/home/trials" in run_cmd
        assert "python:3.11-slim" in run_cmd

        # Verify workspace & git init command executed
        second_call = mock_subprocess_run.call_args_list[1]
        exec_cmd = second_call.args[0]
        assert exec_cmd[:2] == ["docker", "exec"]
        assert "mock_cli_container_abcdef" in exec_cmd
        assert "/bin/bash" in exec_cmd
        assert "git init /home/trials" in exec_cmd[exec_cmd.index("-c") + 1]

    def test_subprocess_start_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=125,
                stdout="",
                stderr="docker daemon not reachable",
            )
            runner = DockerSandboxRunner(backend="subprocess")
            with pytest.raises(RuntimeError, match="Failed to start Docker container: docker daemon not reachable"):
                runner.start()

    def test_subprocess_inject_files(self, mock_subprocess_run):
        runner = DockerSandboxRunner(backend="subprocess")
        runner.start()

        files = {
            "app.py": "print('ok')",
            "data/sample.json": '{"k": "v"}',
        }
        runner.inject_files(files, dest_dir="/home/trials/app")

        # Find the docker cp call
        cp_calls = [
            c for c in mock_subprocess_run.call_args_list
            if len(c.args[0]) >= 2 and c.args[0][1] == "cp"
        ]
        assert len(cp_calls) == 1
        cp_cmd = cp_calls[0].args[0]
        assert cp_cmd == ["docker", "cp", "-", "mock_cli_container_abcdef:/home/trials/app"]

        # Check tar archive input passed into stdin
        tar_input = cp_calls[0].kwargs.get("input")
        assert isinstance(tar_input, bytes)
        buf = io.BytesIO(tar_input)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            assert "app.py" in tar.getnames()
            assert "data/sample.json" in tar.getnames()

    def test_subprocess_run_command_results_and_binary_code(self):
        with patch("subprocess.run") as mock_run:
            # First for run container, second for git init, third for our test command
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="cid123\n", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=2,
                    stdout="some test stdout",
                    stderr="assertion error",
                ),
            ]

            runner = DockerSandboxRunner(backend="subprocess")
            runner.start()

            result = runner.run_command("pytest")
            assert result.command == "pytest"
            assert result.stdout == "some test stdout"
            assert result.stderr == "assertion error"
            assert result.exit_code == 2
            assert result.succeeded is False
            assert result.binary_exit_code == 1

    def test_subprocess_run_command_timeout(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="cid123\n", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5, output=b"partial", stderr=b"hung"),
            ]

            runner = DockerSandboxRunner(backend="subprocess")
            runner.start()

            result = runner.run_command("hang", timeout=5)
            assert result.exit_code == 124
            assert result.binary_exit_code == 1
            assert result.succeeded is False
            assert result.stdout == "partial"
            assert "timed out after 5 seconds" in result.stderr

    def test_subprocess_stop(self, mock_subprocess_run):
        runner = DockerSandboxRunner(backend="subprocess")
        runner.start()
        assert runner.is_running

        runner.stop()
        assert not runner.is_running
        assert runner.container_id is None

        # Verify docker rm -f was called
        rm_calls = [
            c for c in mock_subprocess_run.call_args_list
            if len(c.args[0]) >= 2 and c.args[0][1] == "rm"
        ]
        assert len(rm_calls) == 1
        assert rm_calls[0].args[0] == ["docker", "rm", "-f", "mock_cli_container_abcdef"]


# ============================================================================
# 4. State Validation & Error Guardrails
# ============================================================================


def test_command_and_inject_before_start_raises():
    runner = DockerSandboxRunner(backend="subprocess")
    with pytest.raises(RuntimeError, match="Sandbox container is not running"):
        runner.run_command("ls")

    with pytest.raises(RuntimeError, match="Sandbox container is not running"):
        runner.inject_files({"test.py": "x = 1"})
