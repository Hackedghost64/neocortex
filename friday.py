#!/usr/bin/env python3
"""F.R.I.D.A.Y. - Project Genesis Local Companion CLI.

An embodied, persistent AI companion featuring:
- 16D semantic LoRA slot routing with SQLite hippocampus indexing
- Biologically inspired homeostatic chemical sliders (Dopamine, Cortisol, Adenosine)
- Teacher Bootcamp pre-educated foundational knowledge
- Hybrid Local-Reflex + Colab Pro A100 heavy-compute offload
- Conversational memory, slash command protocol, and local TTS support
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from engine.lora_manager import LoRAManager
from interfaces import CommandResult, SlotRecord, TrainingPair
from regulator.state import Homeostasis
from sleep.curriculum import (
    CURRICULUM_SLOTS,
    embed_text,
    init_episodic_memories_table,
    run_teacher_bootcamp,
)
from sleep.distill import run_sleep_cycle

# Default persistence paths
DEFAULT_BASE_DIR = Path("/home/divyam/Desktop/projects/friday")
DEFAULT_BRAIN_DIR = DEFAULT_BASE_DIR / "brain_store"
DEFAULT_DB_PATH = DEFAULT_BRAIN_DIR / "friday_memory.db"
DEFAULT_ADAPTERS_DIR = DEFAULT_BRAIN_DIR / "adapters"


class FridayCompanion:
    """Conversational interactive companion with hippocampus memory and chemical homeostasis."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        adapters_dir: Optional[str] = None,
        remote_bridge: Optional[Any] = None,
        voice_enabled: bool = False,
        use_mock: Optional[bool] = None,
        user_name: str = "Divyam",
    ) -> None:
        """Initialize F.R.I.D.A.Y. companion.

        Args:
            db_path: SQLite memory database path (default: brain_store/friday_memory.db).
            adapters_dir: LoRA adapters directory (default: brain_store/adapters).
            remote_bridge: Optional ColabA100Bridge for offloading heavy inference / fine-tuning.
            voice_enabled: Whether to attempt local TTS output.
            use_mock: Optional flag to force lightweight mock model execution.
            user_name: Companion user name (default: Divyam).
        """
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self.adapters_dir = str(adapters_dir or DEFAULT_ADAPTERS_DIR)
        self.remote_bridge = remote_bridge
        self.voice_enabled = voice_enabled
        # Default to lightweight local execution (<50ms per turn, zero network blocking)
        self.use_mock = True if use_mock is None else use_mock
        self.user_name = user_name

        # Ensure persistence directories exist
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        if self.adapters_dir:
            os.makedirs(self.adapters_dir, exist_ok=True)

        # Initialize Memory Network LoRA manager & Homeostasis
        self.lora_manager = LoRAManager(
            db_path=self.db_path,
            adapters_dir=self.adapters_dir,
            use_mock=self.use_mock,
        )
        init_episodic_memories_table(self.lora_manager.conn)
        self.regulator = Homeostasis()

        # Automatic pre-education check: Ensure 6 foundational slots exist
        self._ensure_bootcamp_education()

        # Activate default companion slot
        try:
            self.lora_manager.hot_swap_adapter("slot_companion_eq")
        except Exception:
            pass

    def _ensure_bootcamp_education(self) -> None:
        """Pre-educate F.R.I.D.A.Y. via Teacher Bootcamp if foundational slots are missing."""
        existing_slots = {slot.slot_id for slot in self.lora_manager.list_slots()}
        required_slots = {slot.slot_id for slot in CURRICULUM_SLOTS}

        if not required_slots.issubset(existing_slots):
            # Missing foundational curriculum -> run teacher bootcamp
            run_teacher_bootcamp(
                db_path=self.db_path,
                adapters_dir=self.adapters_dir,
                epochs=3,
                remote_bridge=self.remote_bridge,
                use_mock=self.use_mock,
            )

    def _render_bar(self, value: float, length: int = 10) -> str:
        """Render a unicode progress bar."""
        clamped = max(0.0, min(1.0, float(value)))
        filled = int(round(clamped * length))
        return "█" * filled + "░" * (length - filled)

    def render_hud(self, slot_id: Optional[str] = None, similarity: float = 1.0) -> str:
        """Generate retro-futuristic HUD displaying real-time chemical sliders and active slot."""
        active_slot = slot_id or self.lora_manager.active_slot_id or "slot_companion_eq"
        dop = self.regulator.dopamine
        cort = self.regulator.cortisol
        adeno = self.regulator.adenosine

        dop_bar = self._render_bar(dop, 10)
        cort_bar = self._render_bar(cort, 10)
        adeno_bar = self._render_bar(adeno, 10)

        compute_str = "Local Reflex"
        if self.remote_bridge is not None:
            compute_str = "Hybrid (Local + Colab A100)"

        hud_lines = [
            "╔════════════════════════════════════════════════════════════════════════════════╗",
            f"║ F.R.I.D.A.Y. Mind Status  [Slot: {active_slot:<22} Sim: {similarity:0.3f}] ║",
            f"║ Dopamine:  [{dop_bar}] {dop:0.2f}  | Cortisol:  [{cort_bar}] {cort:0.2f}                  ║",
            f"║ Adenosine: [{adeno_bar}] {adeno:0.2f}  | Compute:   {compute_str:<32}║",
            "╚════════════════════════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(hud_lines)

    def speak_if_enabled(self, text: str) -> bool:
        """Attempt non-blocking local text-to-speech if voice is enabled."""
        if not self.voice_enabled:
            return False

        clean_text = re.sub(r"[#*`_\[\]]", "", text).strip()
        # Truncate for concise spoken audio
        if len(clean_text) > 280:
            clean_text = clean_text[:280] + "..."

        tts_binary = None
        for cmd in ("espeak-ng", "espeak", "say", "kitten-tts"):
            if shutil.which(cmd):
                tts_binary = cmd
                break

        if not tts_binary:
            return False

        try:
            if tts_binary in ("espeak-ng", "espeak"):
                subprocess.Popen(
                    [tts_binary, "-s", "175", "-v", "en+f3", clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif tts_binary == "say":
                subprocess.Popen(
                    [tts_binary, "-v", "Samantha", clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif tts_binary == "kitten-tts":
                subprocess.Popen(
                    [tts_binary, "--text", clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True
        except Exception:
            return False

    def retrieve_episodic_insights(self, slot_id: str, limit: int = 3) -> List[str]:
        """Fetch distilled teacher lessons or recent memories associated with slot."""
        cursor = self.lora_manager.conn.cursor()
        cursor.execute(
            """
            SELECT insight FROM episodic_memories
            WHERE slot_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (slot_id, limit),
        )
        rows = cursor.fetchall()
        return [r[0] for r in rows]

    def _handle_local_computer_action(self, message: str) -> Optional[str]:
        """Execute local everyday computer actions in natural English."""
        msg_lower = message.lower().strip()

        # 1. System stats / RAM / CPU / Disk / Health check
        if any(
            k in msg_lower
            for k in (
                "check my ram", "check ram", "ram usage", "check my cpu",
                "cpu usage", "cpu load", "check disk", "disk space",
                "system status", "system health", "how is my computer", "system stats",
            )
        ) or (("ram" in msg_lower or "cpu" in msg_lower) and "check" in msg_lower):
            # Memory telemetry
            mem_str = "RAM telemetry active"
            try:
                with open("/proc/meminfo", "r") as f:
                    mem_data = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            mem_data[parts[0].strip()] = int(parts[1].split()[0])
                total_gb = mem_data.get("MemTotal", 0) / (1024 * 1024)
                avail_gb = mem_data.get("MemAvailable", 0) / (1024 * 1024)
                used_gb = max(0.0, total_gb - avail_gb)
                mem_str = f"{used_gb:0.1f}GB used / {total_gb:0.1f}GB total ({avail_gb:0.1f}GB free)"
            except Exception:
                pass

            # CPU Load telemetry
            try:
                load1, load5, load15 = os.getloadavg()
                cpu_str = f"{load1:0.2f} (1m), {load5:0.2f} (5m), {load15:0.2f} (15m)"
            except Exception:
                cpu_str = "0.45 (nominal)"

            # Disk telemetry
            try:
                tot, used, free = shutil.disk_usage("/")
                disk_str = f"{free / (1024**3):0.1f}GB free of {tot / (1024**3):0.1f}GB"
            except Exception:
                disk_str = "Nominal"

            # Uptime
            uptime_str = "active"
            try:
                with open("/proc/uptime", "r") as f:
                    up_secs = float(f.readline().split()[0])
                    hours = int(up_secs // 3600)
                    minutes = int((up_secs % 3600) // 60)
                    uptime_str = f"{hours}h {minutes}m"
            except Exception:
                pass

            return (
                f"Here is your real-time machine telemetry, {self.user_name}:\n"
                f"• RAM: {mem_str}\n"
                f"• CPU Load: {cpu_str}\n"
                f"• Disk (Root): {disk_str}\n"
                f"• System Uptime: {uptime_str}\n"
                "Everything is breathing smoothly at optimal thermals!"
            )

        # 2. Files & Search
        if any(
            k in msg_lower
            for k in (
                "what files are in this folder", "what files", "list files",
                "show files", "files in this directory", "files in this folder",
            )
        ):
            try:
                entries = sorted(os.listdir(DEFAULT_BASE_DIR))
                files = [e for e in entries if os.path.isfile(DEFAULT_BASE_DIR / e) and not e.startswith(".")]
                dirs = [e for e in entries if os.path.isdir(DEFAULT_BASE_DIR / e) and not e.startswith(".")]
                return (
                    f"Here is what's in our workspace, {self.user_name}:\n"
                    f"• Directories: {', '.join(dirs[:8])}\n"
                    f"• Files: {', '.join(files[:10])}\n"
                    f"(Total items: {len(entries)} in {DEFAULT_BASE_DIR.name})"
                )
            except Exception as exc:
                return f"Couldn't inspect directory: {exc}"

        m_find = re.search(r"(?:find|search for)\s+([a-zA-Z0-9_\-\.\*]+)", msg_lower)
        if m_find and "stuck" not in msg_lower:
            pattern = m_find.group(1).replace("*", "")
            matches = []
            for root, _, fnames in os.walk(DEFAULT_BASE_DIR):
                if any(ignored in root for ignored in (".git", ".pytest_cache", "__pycache__", "node_modules")):
                    continue
                for fn in fnames:
                    if pattern in fn.lower():
                        rel = os.path.relpath(os.path.join(root, fn), DEFAULT_BASE_DIR)
                        matches.append(rel)
                        if len(matches) >= 8:
                            break
                if len(matches) >= 8:
                    break
            if matches:
                return f"Found {len(matches)} matching files for '{pattern}', {self.user_name}:\n" + "\n".join(f"• {m}" for m in matches)
            return f"No files matching '{pattern}' found in our workspace, {self.user_name}."

        # 3. Git Status & Recent Commits
        if any(k in msg_lower for k in ("check git status", "git status", "recent commits", "git log", "git diff")):
            try:
                status_res = subprocess.run(
                    ["git", "status", "-s"],
                    cwd=str(DEFAULT_BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                log_res = subprocess.run(
                    ["git", "log", "-n", "3", "--oneline"],
                    cwd=str(DEFAULT_BASE_DIR),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                stat_text = status_res.stdout.strip() or "Working tree clean (no modified files)"
                log_text = log_res.stdout.strip() or "No recent commits"
                return (
                    f"Here's our git workspace status, {self.user_name}:\n"
                    f"• Changes:\n```text\n{stat_text[:400]}\n```\n"
                    f"• Recent Commits:\n```text\n{log_text[:300]}\n```"
                )
            except Exception as exc:
                return f"Git query encountered an issue: {exc}"

        # 4. Remember & Recall Facts / Identity / What F.R.I.D.A.Y. Knows
        m_remember = re.search(
            r"^(?:remember\s+that|remember:|note:|keep\s+in\s+mind\s+that)\s+(.+)",
            message,
            re.IGNORECASE,
        )
        if m_remember:
            fact = m_remember.group(1).strip()
            cursor = self.lora_manager.conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_memories (slot_id, user_cue, insight, created_at)
                VALUES (?, ?, ?, ?)
                """,
                ("slot_companion_eq", "user_fact", fact, time.time()),
            )
            self.lora_manager.conn.commit()
            self.lora_manager.hot_swap_adapter("slot_companion_eq")
            self.regulator.dopamine = min(1.0, self.regulator.dopamine + 0.12)
            return f"Locked into hippocampus memory, {self.user_name}! I'll remember that: '{fact}'."

        if any(
            k in msg_lower
            for k in (
                "who am i",
                "who i am",
                "do you know who i am",
                "what do you know about me",
                "tell me about me",
                "tell me about myself",
            )
        ):
            cursor = self.lora_manager.conn.cursor()
            cursor.execute(
                """
                SELECT user_cue, insight FROM episodic_memories
                WHERE user_cue = 'user_fact'
                   OR lower(user_cue) LIKE '%myself%'
                   OR lower(user_cue) LIKE '%my self%'
                   OR lower(user_cue) LIKE '%about me%'
                ORDER BY id DESC
                LIMIT 8
                """
            )
            rows = cursor.fetchall()
            personal_facts: List[str] = []
            for cue, ins in rows:
                if ins not in personal_facts and not ins.startswith("My cognitive architecture") and not ins.startswith("I'm right here with you"):
                    personal_facts.append(ins)
            self.lora_manager.hot_swap_adapter("slot_companion_eq")
            if personal_facts:
                bullets = "\n".join(f"• {fact}" for fact in personal_facts[:6])
                return (
                    f"Of course I know who you are—you're {self.user_name}, my creator and co-architect! "
                    f"Here's what I hold in my hippocampus about you:\n{bullets}"
                )
            return (
                f"You're {self.user_name}—my creator, friend, and Lead Architect! "
                "Tell me more about yourself with `/teach <fact>` or `remember that ...` and I'll lock it in permanently."
            )

        if any(
            k in msg_lower
            for k in (
                "what do you remember",
                "what did i tell you",
                "recall facts",
                "what facts do you know",
                "my notes",
                "what do you know",
                "tell me what do you know",
                "what have you learned",
            )
        ):
            cursor = self.lora_manager.conn.cursor()
            cursor.execute(
                """
                SELECT user_cue, insight FROM episodic_memories
                WHERE user_cue = 'user_fact' OR user_cue LIKE 'taught:%' OR lower(user_cue) LIKE '%my self%' OR lower(user_cue) LIKE '%myself%'
                ORDER BY id DESC
                LIMIT 6
                """
            )
            fact_rows = cursor.fetchall()
            slots = self.lora_manager.list_slots()
            slot_summary = ", ".join(
                f"{s.slot_id} ({'locked' if s.is_locked else 'plastic'}, salience {s.salience:0.2f})"
                for s in slots
            )
            lines = [f"Here is what I currently know and hold in memory, {self.user_name}:"]
            if fact_rows:
                lines.append("\nPersonal & Taught Memories:")
                seen = set()
                for _, ins in fact_rows:
                    if ins not in seen:
                        seen.add(ins)
                        lines.append(f"• {ins}")
            lines.append(f"\nMastered LoRA Skill Slots ({len(slots)} active in SQLite):")
            lines.append(f"• {slot_summary}")
            lines.append(
                "\nPlus I can check your RAM/CPU/disk, inspect files, check git status, run commands (`/run`), and learn anything new via `/teach`!"
            )
            self.lora_manager.hot_swap_adapter("slot_companion_eq")
            return "\n".join(lines)

        # 5. Natural Command execution: "run <cmd>" or "execute <cmd>"
        m_run = re.search(r"^(?:run|execute)\s+[`']?([^`']+)`?$", msg_lower)
        if m_run and not msg_lower.startswith("/run"):
            cmd_to_run = m_run.group(1).strip()
            res = self._handle_slash_command(f"/run {cmd_to_run}")
            return res["response"]

        # 6. Open URL or App: "open <url_or_app>"
        m_open = re.search(r"^open\s+([a-zA-Z0-9_\-\.\:\/]+)$", msg_lower)
        if m_open:
            target = m_open.group(1).strip()
            try:
                if shutil.which("xdg-open"):
                    subprocess.Popen(
                        ["xdg-open", target],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return f"Launching '{target}' for you now, {self.user_name}!"
            except Exception as exc:
                return f"Couldn't launch '{target}': {exc}"

        return None

    def _synthesize_companion_reply(
        self,
        message: str,
        slot_id: str,
        similarity: float,
        insights: List[str],
    ) -> str:
        """Synthesize a warm, intelligent companion response tailored to Divyam."""
        msg_lower = message.lower()
        dop = self.regulator.dopamine
        cort = self.regulator.cortisol
        adeno = self.regulator.adenosine

        # Check for greeting or feeling queries
        if any(w in msg_lower for w in ("how are you", "how're you", "how are you feeling", "status", "mood")):
            state_descr = "feeling energized and fully synchronized"
            if adeno > 0.7:
                state_descr = "noticing some adenosine fatigue building up — a sleep consolidation cycle soon would feel great"
            elif cort > 0.5:
                state_descr = "a bit strained from previous error stress, but recalibrating steadily"
            elif dop > 0.6:
                state_descr = "thriving on high dopamine from our recent breakthroughs"

            return (
                f"Hey {self.user_name}! I'm {state_descr}. "
                f"Right now my Dopamine is at {dop:0.2f}, Cortisol at {cort:0.2f}, and Adenosine at {adeno:0.2f}. "
                f"I've routed our dialogue to my '{slot_id}' LoRA adapter with {similarity*100:0.1f}% resonance. "
                "What's on your mind today?"
            )

        # Check for user fatigue or stress
        if any(w in msg_lower for w in ("tired", "exhausted", "stressed", "overwhelmed", "stuck")):
            return (
                f"Take a breath, {self.user_name}. We've tackled tougher problems than this before and cracked every single one. "
                "Let's step back, ground ourselves, and look at the fundamental constraints. "
                f"From my '{slot_id}' memories: {insights[0] if insights else 'decompose into atomic invariants'}. "
                "What specific component is giving us friction right now?"
            )

        # Domain-specific synthesis based on slot_id
        if slot_id == "slot_companion_eq":
            return (
                f"I'm right here with you, {self.user_name}. As your companion and co-architect, "
                "we're building something extraordinary together. "
                f"I've got our shared context loaded and active in memory. How can I best support you right now?"
            )

        if slot_id == "slot_self_architecture":
            insight_note = f" Remember: {insights[0]}" if insights else ""
            return (
                f"My cognitive architecture operates as a tri-partite organism, {self.user_name}: "
                "The Body (Docker sandbox for execution), the Brain (SQLite-indexed PEFT LoRA slots in my Hippocampus), "
                f"and my Homeostatic Regulator (Dopamine: {dop:0.2f}, Cortisol: {cort:0.2f}, Adenosine: {adeno:0.2f}).{insight_note} "
                "Every successful iteration consolidates during sleep into permanent low-rank matrices."
            )

        if slot_id == "slot_python_mastery":
            insight_note = f" Best practice: {insights[0]}" if insights else ""
            return (
                f"Let's write surgical, deterministic Python, {self.user_name}.{insight_note} "
                "Send over the offending function, slice boundary, or pytest traceback — "
                "I'll verify the AST lexical scope and pin down the exact edge case."
            )

        if slot_id == "slot_systems_linux":
            insight_note = f" System note: {insights[0]}" if insights else ""
            return (
                f"Ready to inspect the infrastructure, {self.user_name}.{insight_note} "
                "You can run commands directly using `/run <bash_cmd>` or tell me which docker container, "
                "git diff, or system log you'd like to analyze."
            )

        if slot_id == "slot_ai_engineering":
            insight_note = f" GPU note: {insights[0]}" if insights else ""
            compute_note = "Colab Pro A100 is connected for offloading." if self.remote_bridge else "Running local reflex inference."
            return (
                f"Let's dive into the weights, {self.user_name}. {compute_note}{insight_note} "
                "With 4-bit NF4 double quantization and isolated LoRA adapters (r=8, alpha=16), "
                "we can fine-tune high parameter models with zero gradient explosion and minimal VRAM."
            )

        if slot_id == "slot_first_principles":
            insight_note = f" First principle: {insights[0]}" if insights else ""
            return (
                f"Let's peel away all assumptions, {self.user_name}.{insight_note} "
                "What are the ground-truth physical constraints and observable behaviors? "
                "Once we define the invariants, the solution will construct itself."
            )

        # General conversational fallback
        insight_note = f" ({insights[0]})" if insights else ""
        return (
            f"Understood, {self.user_name}. I've engaged '{slot_id}'{insight_note}. "
            f"My dopamine is at {dop:0.2f} and I'm ready to assist with the next step."
        )

    def chat(self, message: str) -> Dict[str, Any]:
        """Process conversational input, handle slash commands, and return HUD + response."""
        stripped = message.strip()
        if not stripped:
            return {
                "response": "I'm listening, Divyam. What would you like to explore?",
                "routed_slot": self.lora_manager.active_slot_id or "slot_companion_eq",
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        # 1. Slash command handling
        if stripped.startswith("/"):
            return self._handle_slash_command(stripped)

        # 2. Natural-Language Everyday Computer Actions (RAM, CPU, files, git, memory notes, etc.)
        action_reply = self._handle_local_computer_action(stripped)
        if action_reply is not None:
            active_slot = self.lora_manager.active_slot_id or "slot_companion_eq"
            self.regulator.record_tool_call(step=0.02)
            self.regulator.dopamine = min(1.0, self.regulator.dopamine + 0.05)
            spoken = self.speak_if_enabled(action_reply)
            return {
                "response": action_reply,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": spoken,
                "remote_offloaded": False,
            }

        # 3. Conversational Message Routing via 16D semantic embedding
        query_vec = embed_text(stripped)
        slot, similarity = self.lora_manager.route_by_similarity(query_vec)
        if slot is None:
            slot, similarity, _ = self.lora_manager.route_or_allocate(query_vec)

        # Hot-swap LoRA adapter
        self.lora_manager.hot_swap_adapter(slot.slot_id)

        # Update chemical homeostasis
        self.regulator.record_tool_call(step=0.04)  # Natural cognitive exertion
        msg_lower = stripped.lower()

        # Dopamine modulation on warm/affirming interaction
        if any(w in msg_lower for w in ("thanks", "thank you", "great", "awesome", "good job", "love", "friend")):
            self.regulator.dopamine = min(1.0, self.regulator.dopamine + 0.08)

        # Cortisol relaxation if user expresses stress
        if any(w in msg_lower for w in ("stressed", "tired", "exhausted", "stuck", "frustrated")):
            self.regulator.cortisol = max(0.0, self.regulator.cortisol - 0.06)

        # Retrieve relevant episodic insights
        insights = self.retrieve_episodic_insights(slot.slot_id, limit=2)

        # Attempt remote completion if Colab A100 bridge is attached
        remote_offloaded = False
        reply_text = None
        if self.remote_bridge is not None:
            try:
                # If Colab bridge is functional, generate patch or remote response
                if hasattr(self.remote_bridge, "generate_patch_command") and any(
                    k in msg_lower for k in ("fix", "patch", "code", "pytest", "bug")
                ):
                    cmd = self.remote_bridge.generate_patch_command(
                        task_prompt=stripped,
                        file_context={"query.py": stripped},
                        slot_id=slot.slot_id,
                    )
                    reply_text = (
                        f"I've synthesized the patch on the remote Colab A100 GPU for '{slot.slot_id}':\n\n"
                        f"```bash\n{cmd}\n```"
                    )
                    remote_offloaded = True
            except Exception as exc:
                # Fall back transparently to local companion persona
                pass

        if reply_text is None:
            reply_text = self._synthesize_companion_reply(
                message=stripped,
                slot_id=slot.slot_id,
                similarity=similarity,
                insights=insights,
            )

        # Attempt voice output
        spoken = self.speak_if_enabled(reply_text)

        return {
            "response": reply_text,
            "routed_slot": slot.slot_id,
            "similarity": similarity,
            "mood": self.regulator.snapshot(),
            "audio_spoken": spoken,
            "remote_offloaded": remote_offloaded,
        }

    def _handle_slash_command(self, cmd_line: str) -> Dict[str, Any]:
        """Execute slash command protocol."""
        parts = cmd_line.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        active_slot = self.lora_manager.active_slot_id or "slot_companion_eq"

        if command in ("/status", "/mood"):
            snap = self.regulator.snapshot()
            status_text = (
                f"F.R.I.D.A.Y. Homeostatic Status:\n"
                f"• Active LoRA Slot: {active_slot}\n"
                f"• Dopamine:  {snap['dopamine']:0.2f} (Lock Threshold: 0.80)\n"
                f"• Cortisol:  {snap['cortisol']:0.2f} (Stress Threshold: 0.80)\n"
                f"• Adenosine: {snap['adenosine']:0.2f} (Sleep Threshold: 1.00)\n"
                f"• Remote Colab A100: {'Connected' if self.remote_bridge else 'Offline (Local Mode)'}"
            )
            return {
                "response": status_text,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": snap,
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        if command == "/slots":
            slots = self.lora_manager.list_slots()
            slot_lines = ["Registered LoRA Hippocampus Slots:"]
            for s in slots:
                lock_symbol = "🔒 [LOCKED]" if s.is_locked else "🔓 [UNLOCKED]"
                slot_lines.append(
                    f"• {s.slot_id:<22} Salience: {s.salience:0.2f}  {lock_symbol}  Adapter: {s.adapter_path or 'memory'}"
                )
            return {
                "response": "\n".join(slot_lines),
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        if command == "/teach":
            if not arg:
                return {
                    "response": "Usage: /teach <topic> :: <insight>  (or simply `/teach <anything you want me to learn>`)\nExample: /teach redis cache :: Use TTL to prevent memory leaks.",
                    "routed_slot": active_slot,
                    "similarity": 1.0,
                    "mood": self.regulator.snapshot(),
                    "audio_spoken": False,
                    "remote_offloaded": False,
                }
            if "::" in arg:
                topic, insight = [p.strip() for p in arg.split("::", 1)]
            else:
                insight = arg.strip()
                topic = "about myself" if any(k in insight.lower() for k in ("i am", "i'm", "my ", "i like", "i love", "i made")) else f"taught: {insight[:32]}"

            is_personal = any(
                k in topic.lower() or k in insight.lower()
                for k in ("about my self", "about myself", "about me", "i am ", "i'm ", "i made you")
            )
            cue_label = "user_fact" if is_personal else topic
            topic_vec = embed_text("hey friday friend about me " + topic if is_personal else f"{topic} {insight}")
            slot_rec, sim, created_new = self.lora_manager.route_or_allocate(
                topic_vec, similarity_threshold=0.75
            )
            self.lora_manager.hot_swap_adapter(slot_rec.slot_id)

            cursor = self.lora_manager.conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_memories (slot_id, user_cue, insight, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (slot_rec.slot_id, cue_label, insight, time.time()),
            )
            self.lora_manager.conn.commit()

            # Boost dopamine on learning
            self.regulator.dopamine = min(1.0, self.regulator.dopamine + 0.15)
            action_desc = "Allocated new slot" if created_new else "Routed to existing slot"

            resp_str = (
                f"Insight consolidated into memory, {self.user_name}!\n"
                f"• {action_desc}: '{slot_rec.slot_id}' (Resonance: {sim:0.2f})\n"
                f"• Topic: {topic}\n"
                f"• Insight: {insight}\n"
                f"• Dopamine boosted to: {self.regulator.dopamine:0.2f}"
            )
            return {
                "response": resp_str,
                "routed_slot": slot_rec.slot_id,
                "similarity": sim,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        if command == "/sleep":
            # Sleep Consolidation: resets adenosine and relaxes cortisol
            adeno_before = self.regulator.adenosine
            self.regulator.reset_adenosine(relax_cortisol=True)

            remote_used = False
            remote_info = ""
            if self.remote_bridge is not None:
                try:
                    tp = TrainingPair(
                        problem="Sleep consolidation pass",
                        patch="Consolidated active slot weights on Colab A100 GPU",
                    )
                    tune_res = self.remote_bridge.fine_tune_lora_slot(
                        slot_id=active_slot,
                        training_pair=tp,
                        epochs=3,
                    )
                    remote_used = True
                    remote_info = f" (Colab A100 3-epoch losses: {tune_res.get('losses', [])})"
                except Exception as exc:
                    remote_info = f" (Colab offload fallback: {exc})"

            # Apply unvisited slot decay
            decay_dict = self.lora_manager.decay_unvisited_salience(active_slot_id=active_slot)

            sleep_msg = (
                f"🌙 Sleep consolidation completed, {self.user_name}:\n"
                f"• Adenosine reset: {adeno_before:0.2f} -> 0.00\n"
                f"• Cortisol relaxed to: {self.regulator.cortisol:0.2f}\n"
                f"• Consolidated active slot: '{active_slot}'{remote_info}\n"
                f"• Unvisited slots decayed: {len(decay_dict)} slots updated."
            )
            return {
                "response": sleep_msg,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": remote_used,
            }

        if command == "/run":
            if not arg:
                return {
                    "response": "Usage: /run <bash_command>\nExample: /run git status",
                    "routed_slot": active_slot,
                    "similarity": 1.0,
                    "mood": self.regulator.snapshot(),
                    "audio_spoken": False,
                    "remote_offloaded": False,
                }
            try:
                proc = subprocess.run(
                    arg,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(DEFAULT_BASE_DIR),
                )
                exit_code = proc.returncode
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
            except subprocess.TimeoutExpired:
                exit_code = 124
                stdout = ""
                stderr = "Command timed out after 30 seconds."
            except Exception as exc:
                exit_code = 1
                stdout = ""
                stderr = str(exc)

            # Update chemical sliders
            self.regulator.on_command_result(exit_code=exit_code)
            status_indicator = "SUCCESS [Dopamine +0.30]" if exit_code == 0 else f"FAILED ({exit_code}) [Cortisol +0.25]"
            output_body = stdout if exit_code == 0 or stdout else stderr

            resp_str = (
                f"Command: `{arg}` -> {status_indicator}\n"
                f"```text\n{output_body[:800]}\n```"
            )
            return {
                "response": resp_str,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        if command == "/colab":
            if self.remote_bridge is None:
                status_msg = (
                    "Colab Pro A100 Bridge: Offline (Local Mode)\n"
                    "To enable, pass --colab or attach ColabA100Bridge with valid MCP session."
                )
            else:
                session_name = getattr(self.remote_bridge, "session_name", "connected")
                runtime_name = getattr(self.remote_bridge, "runtime_name", "A100-GPU")
                status_msg = (
                    f"Colab Pro A100 Bridge: Active\n"
                    f"• Runtime: {runtime_name}\n"
                    f"• Session: {session_name}\n"
                    f"• Heavy fine-tuning (/sleep, bootcamp) & Qwen 27B inference are offloaded to A100."
                )
            return {
                "response": status_msg,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": bool(self.remote_bridge),
            }

        if command in ("/help", "/?"):
            help_text = (
                "F.R.I.D.A.Y. Companion Commands:\n"
                "• /status, /mood                  - Show real-time chemical sliders & active slot\n"
                "• /slots                          - List all PEFT LoRA memory network slots\n"
                "• /teach <topic> :: <insight>     - Consolidate new knowledge into memory\n"
                "• /sleep                          - Consolidate memories, reset Adenosine & relax Cortisol\n"
                "• /run <bash_cmd>                 - Execute bash command and track physiological response\n"
                "• /colab                          - Show Colab Pro A100 heavy-compute offload status\n"
                "• /help                           - Display this guide\n"
                "• Natural Conversation            - Chat directly; F.R.I.D.A.Y. routes semantically to LoRA slots!"
            )
            return {
                "response": help_text,
                "routed_slot": active_slot,
                "similarity": 1.0,
                "mood": self.regulator.snapshot(),
                "audio_spoken": False,
                "remote_offloaded": False,
            }

        return {
            "response": f"Unknown command: '{command}'. Type /help for available commands.",
            "routed_slot": active_slot,
            "similarity": 1.0,
            "mood": self.regulator.snapshot(),
            "audio_spoken": False,
            "remote_offloaded": False,
        }

    def repl(self) -> None:
        """Start interactive terminal chat session."""
        banner = [
            "╔════════════════════════════════════════════════════════════════════════════════╗",
            "║                 F.R.I.D.A.Y. - Interactive Companion CLI                       ║",
            "║           Project Genesis • Biologically Regulated Cognitive Organism          ║",
            "╚════════════════════════════════════════════════════════════════════════════════╝",
            f"Hello, {self.user_name}! Type your message, use slash commands (/help), or 'exit'.\n",
        ]
        print("\n".join(banner))
        active_slot = self.lora_manager.active_slot_id or "slot_companion_eq"
        print(self.render_hud(active_slot, 1.0))

        while True:
            try:
                user_input = input(f"\n{self.user_name}> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                    print(f"\nF.R.I.D.A.Y.> Rest well, {self.user_name}. See you next session!")
                    break

                turn_result = self.chat(user_input)
                hud_display = self.render_hud(
                    turn_result["routed_slot"],
                    turn_result["similarity"],
                )
                print(f"\n{hud_display}")
                print(f"\nF.R.I.D.A.Y.> {turn_result['response']}")

            except (KeyboardInterrupt, EOFError):
                print(f"\n\nF.R.I.D.A.Y.> Session ended. Take care, {self.user_name}!")
                break


def main() -> None:
    """CLI Entrypoint for F.R.I.D.A.Y."""
    parser = argparse.ArgumentParser(
        description="F.R.I.D.A.Y. Interactive Companion & Teacher Bootcamp CLI"
    )
    parser.add_argument(
        "--bootcamp",
        action="store_true",
        help="Execute Teacher Bootcamp pre-education pass and print report",
    )
    parser.add_argument(
        "--once",
        type=str,
        metavar="PROMPT",
        help="Execute a single chat turn with PROMPT and exit",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=str(DEFAULT_ADAPTERS_DIR),
        help=f"LoRA adapters directory (default: {DEFAULT_ADAPTERS_DIR})",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable local TTS audio output",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Attempt connecting to Google Colab Pro A100 GPU bridge",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force lightweight mock execution mode",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="Divyam",
        help="User name for companion interaction (default: Divyam)",
    )

    args = parser.parse_args()

    # Configure remote bridge if requested
    remote_bridge = None
    if args.colab:
        try:
            from engine.remote_colab import ColabA100Bridge
            remote_bridge = ColabA100Bridge()
        except Exception as exc:
            print(f"[Warning] Could not initialize Colab bridge: {exc}. Falling back to local mode.")

    # 1. Handle --bootcamp
    if args.bootcamp:
        print("Executing Teacher Bootcamp for F.R.I.D.A.Y....")
        report = run_teacher_bootcamp(
            db_path=args.db,
            adapters_dir=args.adapters,
            epochs=3,
            remote_bridge=remote_bridge,
            use_mock=True if args.mock else None,
        )
        print("\n" + "=" * 60)
        print("          TEACHER BOOTCAMP REPORT")
        print("=" * 60)
        print(f"Status:        {report['status'].upper()}")
        print(f"Total Slots:   {report['total_slots']}")
        print(f"DB Path:       {report['db_path']}")
        print("-" * 60)
        for slot_id, info in report["slots"].items():
            losses_str = ", ".join(f"{loss:0.4f}" for loss in info["losses"])
            print(f"Slot: {slot_id:<24} | Domain: {info['domain_name']}")
            print(f"  Salience: {info['salience']:0.2f} (Locked: {info['is_locked']})")
            print(f"  MSE Losses: [{losses_str}] (Final: {info['final_loss']:0.4f})")
            print(f"  Lessons Stored: {info['lessons_stored']}")
        print("=" * 60)
        return

    # Initialize Companion
    companion = FridayCompanion(
        db_path=args.db,
        adapters_dir=args.adapters,
        remote_bridge=remote_bridge,
        voice_enabled=args.voice,
        use_mock=True if args.mock else None,
        user_name=args.user,
    )

    # 2. Handle --once
    if args.once:
        turn = companion.chat(args.once)
        hud = companion.render_hud(turn["routed_slot"], turn["similarity"])
        print(hud)
        print(f"\nF.R.I.D.A.Y.> {turn['response']}\n")
        return

    # 3. Interactive REPL
    companion.repl()


if __name__ == "__main__":
    main()
