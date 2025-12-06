# PyTorch 2.8+ compatibility: Add safe globals for model loading
# Must be done before any pyannote/diart imports
import torch
from torch import torch_version

torch.serialization.add_safe_globals([torch_version.TorchVersion])

try:
    from pyannote.audio.core.task import Specifications, Problem, Resolution

    torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
except ImportError:
    pass  # pyannote.audio not installed

from .audio_processor import AudioProcessor
from .core import TranscriptionEngine
from .parse_args import parse_args
from .web.web_interface import get_inline_ui_html, get_web_interface_html

__all__ = [
    "TranscriptionEngine",
    "AudioProcessor",
    "parse_args",
    "get_web_interface_html",
    "get_inline_ui_html",
    "download_simulstreaming_backend",
]
