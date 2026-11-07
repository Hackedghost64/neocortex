"""Engine module for Project Genesis: LoRA memory network and frozen base model."""

from engine.lora_manager import LoRAManager
from engine.remote_colab import ColabA100Bridge

__all__ = ["LoRAManager", "ColabA100Bridge"]
