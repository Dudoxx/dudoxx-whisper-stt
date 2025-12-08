# PyTorch 2.8+ compatibility: Patch torch.load for pyannote/diart model loading
# PyTorch 2.6+ changed default weights_only=True which breaks loading pyannote checkpoints
# These are trusted HuggingFace models, so we allow unsafe loading
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for pyannote compatibility."""
    # Force weights_only=False if not explicitly set to True
    # This handles both missing key and None value cases
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

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
