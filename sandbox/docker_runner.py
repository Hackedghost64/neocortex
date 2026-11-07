"""Docker Execution Sandbox implementation for Project Genesis."""

from __future__ import annotations

import concurrent.futures
import io
import subprocess
import tarfile
import time
from typing import Any, Dict, Optional, Union

try:
    import docker  # type: ignore
    from docker.errors import DockerException  # type: ignore

    HAS_DOCKER_SDK = True
except ImportError:
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore
    HAS_DOCKER_SDK = False

from interfaces import CommandResult, SandboxRunnerProtocol


class DockerSandboxRunner:
    """Isolated Docker container runner implementing SandboxRunnerProtocol.

    Supports both the Docker Python SDK (if installed and daemon responsive)
    and CLI subprocess fallback.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        workspace: str = "/home/trials",
        container_name: Optional[str] = None,
        network_disabled: bool = False,
        backend: str = "auto",
        client: Optional[Any] = None,
    ) -> None:
        self.image = image
        self.workspace = workspace
        self.container_name = container_name
        self.network_disabled = network_disabled

        self._container: Any = None
        self._container_id: Optional[str] = None

        if client is not None:
            self._client = client
            self._backend = "sdk"
        elif backend == "auto":
            if HAS_DOCKER_SDK:
                try:
                    self._client = docker.from_env()
                    self._client.ping()
                    self._backend = "sdk"
                except Exception:
                    self._client = None
                    self._backend = "subprocess"
            else:
                self._client = None
                self._backend = "subprocess"
        elif backend == "sdk":
            if HAS_DOCKER_SDK:
                self._client = docker.from_env()
            else:
                raise RuntimeError("Docker SDK backend requested but 'docker' package is not installed.")
            self._backend = "sdk"
        elif backend in ("subprocess", "cli"):
            self._client = None
            self._backend = "subprocess"
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'auto', 'sdk', 'subprocess'.")

    @property
    def container_id(self) -> Optional[str]:
        """Return the running container ID, or None if not started."""
        return self._container_id

    @property
    def is_running(self) -> bool:
        """Return True if container is active and initialized."""
        return self._container_id is not None

    @property
    def backend(self) -> str:
        """Return current execution backend: 'sdk' or 'subprocess'."""
        return self._backend

    def start(self) -> str:
        """Spin up isolated Docker container and ensure workspace + git init.

        Returns:
            The string ID of the running container.
        """
        if self.is_running and self._container_id:
            return self._container_id

        if self._backend == "sdk":
            self._start_sdk()
        else:
            self._start_subprocess()

        self._ensure_workspace_initialized()
        assert self._container_id is not None
        return self._container_id

    def _start_sdk(self) -> None:
        if self._client is None:
            if not HAS_DOCKER_SDK:
                raise RuntimeError("Docker SDK backend requested but 'docker' package is not installed.")
            self._client = docker.from_env()

        container = self._client.containers.run(
            image=self.image,
            command=["sleep", "infinity"],
            detach=True,
            working_dir=self.workspace,
            name=self.container_name,
            network_disabled=self.network_disabled,
        )
        self._container = container
        self._container_id = getattr(container, "id", str(container))

    def _start_subprocess(self) -> None:
        cmd = ["docker", "run", "-d", "--workdir", self.workspace]
        if self.container_name:
            cmd.extend(["--name", self.container_name])
        if self.network_disabled:
            cmd.extend(["--network", "none"])
        cmd.extend([self.image, "sleep", "infinity"])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as err:
            raise RuntimeError(f"Docker executable not found: {err}") from err

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() if proc.stderr else f"Exit code {proc.returncode}"
            raise RuntimeError(f"Failed to start Docker container: {err_msg}")

        self._container_id = proc.stdout.strip()

    def _ensure_workspace_initialized(self) -> None:
        """Ensure /home/trials exists and is initialized as a safe git repo."""
        init_cmd = (
            f"mkdir -p {self.workspace} && "
            f"if command -v git >/dev/null 2>&1; then "
            f"  git config --system --add safe.directory {self.workspace} 2>/dev/null || "
            f"  git config --global --add safe.directory {self.workspace} 2>/dev/null || true; "
            f"  if ! git -C {self.workspace} rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
            f"    git init {self.workspace} 2>/dev/null || true; "
            f"    git -C {self.workspace} config user.name 'Project Genesis Sandbox' 2>/dev/null || true; "
            f"    git -C {self.workspace} config user.email 'sandbox@genesis.local' 2>/dev/null || true; "
            f"  fi; "
            f"fi"
        )
        self.run_command(init_cmd, workdir="/")

    @staticmethod
    def _build_tar_archive(files: Dict[str, Union[str, bytes]]) -> bytes:
        """Pack in-memory files into a tar byte stream with preserved directory structure."""
        tar_buffer = io.BytesIO()
        dirs_added = set()
        now = int(time.time())

        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for filepath, content in files.items():
                clean_name = filepath.lstrip("/")

                # Pre-create directory headers if nested
                parts = clean_name.split("/")
                for i in range(1, len(parts)):
                    dir_path = "/".join(parts[:i])
                    if dir_path and dir_path not in dirs_added:
                        di = tarfile.TarInfo(name=dir_path)
                        di.type = tarfile.DIRTYPE
                        di.mode = 0o755
                        di.mtime = now
                        tar.addfile(di)
                        dirs_added.add(dir_path)

                data = content.encode("utf-8") if isinstance(content, str) else content
                ti = tarfile.TarInfo(name=clean_name)
                ti.size = len(data)
                ti.mtime = now
                ti.mode = 0o644
                tar.addfile(ti, io.BytesIO(data))

        return tar_buffer.getvalue()

    def inject_files(self, files: Dict[str, Union[str, bytes]], dest_dir: str = "/home/trials") -> None:
        """Pack files into in-memory tar archive and inject into dest_dir inside the container."""
        if not self.is_running:
            raise RuntimeError("Sandbox container is not running. Call start() before injecting files.")

        # Ensure destination directory exists
        self.run_command(f"mkdir -p {dest_dir}", workdir="/")

        tar_bytes = self._build_tar_archive(files)

        if self._backend == "sdk":
            self._container.put_archive(dest_dir, tar_bytes)
        else:
            cmd = ["docker", "cp", "-", f"{self._container_id}:{dest_dir}"]
            proc = subprocess.run(
                cmd,
                input=tar_bytes,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Failed to inject files via docker cp: {err}")

    def run_command(self, command: str, workdir: str = "/home/trials", timeout: int = 60) -> CommandResult:
        """Execute bash command inside the container and capture structured result."""
        if not self.is_running:
            raise RuntimeError("Sandbox container is not running. Call start() before executing commands.")

        if self._backend == "sdk":
            return self._run_command_sdk(command=command, workdir=workdir, timeout=timeout)
        else:
            return self._run_command_subprocess(command=command, workdir=workdir, timeout=timeout)

    def _run_command_sdk(self, command: str, workdir: str, timeout: int) -> CommandResult:
        def _exec() -> Any:
            return self._container.exec_run(
                cmd=["/bin/bash", "-c", command],
                workdir=workdir,
                demux=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_exec)
            try:
                exec_res = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return CommandResult(
                    command=command,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    exit_code=124,
                )

        stdout = ""
        stderr = ""
        if isinstance(exec_res.output, tuple):
            out_b, err_b = exec_res.output
            if out_b:
                stdout = out_b.decode("utf-8", errors="replace")
            if err_b:
                stderr = err_b.decode("utf-8", errors="replace")
        elif isinstance(exec_res.output, (bytes, bytearray)):
            stdout = exec_res.output.decode("utf-8", errors="replace")
        elif exec_res.output:
            stdout = str(exec_res.output)

        exit_code = exec_res.exit_code if exec_res.exit_code is not None else 0
        return CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    def _run_command_subprocess(self, command: str, workdir: str, timeout: int) -> CommandResult:
        cmd = [
            "docker", "exec",
            "-w", workdir,
            str(self._container_id),
            "/bin/bash", "-c", command,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )
            raw_out = proc.stdout
            raw_err = proc.stderr
            stdout = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, (bytes, bytearray)) else str(raw_out or "")
            stderr = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, (bytes, bytearray)) else str(raw_err or "")
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            raw_out = getattr(exc, "stdout", None) or getattr(exc, "output", None)
            raw_err = getattr(exc, "stderr", None)
            stdout = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, (bytes, bytearray)) else str(raw_out or "")
            err_msg = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, (bytes, bytearray)) else str(raw_err or "")
            timeout_msg = f"Command timed out after {timeout} seconds"
            stderr = f"{err_msg}\n{timeout_msg}".strip() if err_msg else timeout_msg
            exit_code = 124

        return CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    def stop(self) -> None:
        """Teardown and remove the container deterministically."""
        if self._backend == "sdk":
            if self._container is not None:
                try:
                    self._container.stop(timeout=5)
                except Exception:
                    pass
                try:
                    self._container.remove(force=True)
                except Exception:
                    pass
                self._container = None
        else:
            if self._container_id is not None:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", str(self._container_id)],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                except Exception:
                    pass

        self._container_id = None

    def __enter__(self) -> DockerSandboxRunner:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
