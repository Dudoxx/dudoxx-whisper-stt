#!/usr/bin/env python3
"""
ASR Enhancement Models for Voxtral Streaming Server

Features:
- Silero VAD: Neural voice activity detection
- Pyannote Speaker Diarization: Who spoke when
- Speaker Embeddings: Track speakers across session

License: Apache 2.0
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global model instances (lazy-loaded)
_silero_vad_model = None
_diarization_pipeline = None
_embedding_model = None


# =============================================================================
# Silero VAD - Neural Voice Activity Detection
# =============================================================================

def get_silero_vad():
    """Lazy-load Silero VAD model."""
    global _silero_vad_model
    if _silero_vad_model is None:
        try:
            from silero_vad import load_silero_vad
            _silero_vad_model = load_silero_vad(onnx=True)  # ONNX for speed
            logger.info("Silero VAD model loaded (ONNX)")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            _silero_vad_model = False
    return _silero_vad_model if _silero_vad_model else None


@dataclass
class VADSegment:
    """Voice activity segment."""
    start: float
    end: float
    confidence: float = 1.0


class SileroVAD:
    """
    Neural Voice Activity Detection using Silero VAD.
    Much more accurate than RMS energy-based detection.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = get_silero_vad()
        self._state = None

    def reset(self):
        """Reset VAD state for new session."""
        self._state = None

    def detect(self, audio: np.ndarray) -> list[VADSegment]:
        """
        Detect speech segments in audio.

        Args:
            audio: Float32 audio samples normalized to [-1, 1]

        Returns:
            List of VADSegment with start/end times
        """
        if self.model is None:
            return []

        try:
            from silero_vad import get_speech_timestamps

            # Ensure correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio)

            # Get speech timestamps
            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )

            # Convert to segments
            segments = []
            for ts in timestamps:
                start_sec = ts["start"] / self.sample_rate
                end_sec = ts["end"] / self.sample_rate
                segments.append(VADSegment(start=start_sec, end=end_sec))

            return segments

        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return []

    def is_speech(self, audio: np.ndarray, min_speech_ratio: float = 0.3) -> bool:
        """Check if audio chunk contains sufficient speech."""
        segments = self.detect(audio)
        if not segments:
            return False

        total_speech = sum(s.end - s.start for s in segments)
        total_duration = len(audio) / self.sample_rate
        return (total_speech / total_duration) >= min_speech_ratio


# =============================================================================
# Speaker Diarization - Who spoke when
# =============================================================================

def get_diarization_pipeline(hf_token: Optional[str] = None):
    """
    Lazy-load pyannote speaker diarization pipeline.
    Requires Hugging Face token with pyannote access.
    """
    global _diarization_pipeline
    if _diarization_pipeline is None:
        try:
            from pyannote.audio import Pipeline

            # Try loading with token (pyannote 3.x uses 'token' not 'use_auth_token')
            if hf_token:
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token,
                )
            else:
                # Try without token (may fail if model requires auth)
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                )

            # Move to GPU if available
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))

            logger.info("Pyannote diarization pipeline loaded on GPU")

        except Exception as e:
            logger.warning(f"Failed to load diarization pipeline: {e}")
            _diarization_pipeline = False

    return _diarization_pipeline if _diarization_pipeline else None


@dataclass
class SpeakerSegment:
    """Speaker segment with timing."""
    speaker: str
    start: float
    end: float
    confidence: float = 0.0


@dataclass
class SpeakerInfo:
    """Information about a detected speaker."""
    id: str
    label: str  # "Speaker 1", "Speaker 2", etc.
    total_duration: float = 0.0
    segment_count: int = 0
    embedding: Optional[np.ndarray] = None


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.
    Identifies different speakers and tracks them across the session.

    Optimizations:
    - min_speakers/max_speakers bounds for better accuracy
    - Exclusive speaker mode for cleaner transcript alignment
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 6,
    ):
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.speakers: dict[str, SpeakerInfo] = {}
        self.speaker_count = 0
        self.sample_rate = 16000
        # Use global pipeline (preloaded at startup)
        self.pipeline = get_diarization_pipeline(hf_token)
        if self.pipeline:
            logger.info(f"SpeakerDiarizer initialized (min={min_speakers}, max={max_speakers})")
        else:
            logger.warning("SpeakerDiarizer: pipeline not available")

    def _ensure_pipeline(self):
        """Ensure pipeline is loaded."""
        if self.pipeline is None:
            self.pipeline = get_diarization_pipeline(self.hf_token)
        return self.pipeline is not None

    def diarize(
        self,
        audio: np.ndarray,
        start_offset: float = 0.0,
    ) -> list[SpeakerSegment]:
        """
        Perform speaker diarization on audio chunk.

        Args:
            audio: Float32 audio samples
            start_offset: Offset in seconds for timestamp alignment

        Returns:
            List of SpeakerSegment with speaker labels and timing
        """
        if not self._ensure_pipeline():
            logger.debug("Diarization skipped: pipeline not available")
            return []

        try:
            # Prepare audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0

            audio_duration = len(audio) / self.sample_rate
            logger.debug(f"Diarizing {audio_duration:.2f}s audio, offset={start_offset:.2f}s")

            # Create waveform dict for pyannote
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
            audio_dict = {"waveform": waveform, "sample_rate": self.sample_rate}

            # Run diarization with optimized parameters
            diarization_result = self.pipeline(
                audio_dict,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

            # pyannote 4.x returns DiarizeOutput, need to access speaker_diarization attribute
            # Prefer exclusive_speaker_diarization for cleaner transcript alignment
            if hasattr(diarization_result, 'exclusive_speaker_diarization'):
                diarization = diarization_result.exclusive_speaker_diarization
            elif hasattr(diarization_result, 'speaker_diarization'):
                diarization = diarization_result.speaker_diarization
            else:
                # Fallback for older pyannote versions
                diarization = diarization_result

            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Track speaker
                if speaker not in self.speakers:
                    self.speaker_count += 1
                    self.speakers[speaker] = SpeakerInfo(
                        id=speaker,
                        label=f"Speaker {self.speaker_count}",
                    )

                speaker_info = self.speakers[speaker]
                speaker_info.total_duration += turn.end - turn.start
                speaker_info.segment_count += 1

                segments.append(SpeakerSegment(
                    speaker=speaker_info.label,
                    start=round(turn.start + start_offset, 3),
                    end=round(turn.end + start_offset, 3),
                ))

            logger.debug(f"Diarization found {len(segments)} segments, {len(self.speakers)} speakers")
            return segments

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return []

    def get_speakers(self) -> list[dict]:
        """Get list of detected speakers with stats."""
        return [
            {
                "id": s.id,
                "label": s.label,
                "total_duration": round(s.total_duration, 2),
                "segment_count": s.segment_count,
            }
            for s in self.speakers.values()
        ]

    def reset(self):
        """Reset for new session."""
        self.speakers.clear()
        self.speaker_count = 0


# =============================================================================
# Combined ASR Enhancer
# =============================================================================

@dataclass
class EnhancedASRResult:
    """Combined result from all ASR enhancers."""
    # VAD
    vad_segments: list[VADSegment] = field(default_factory=list)
    is_speech: bool = True
    speech_ratio: float = 1.0

    # Diarization
    speaker_segments: list[SpeakerSegment] = field(default_factory=list)
    current_speaker: Optional[str] = None
    speakers: list[dict] = field(default_factory=list)

    # Processing time
    vad_time_ms: float = 0.0
    diarization_time_ms: float = 0.0


class ASREnhancer:
    """
    Combined ASR enhancement with VAD and diarization.
    """

    def __init__(
        self,
        enable_vad: bool = True,
        enable_diarization: bool = True,
        hf_token: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.enable_vad = enable_vad
        self.enable_diarization = enable_diarization
        self.sample_rate = sample_rate

        # Initialize components
        self.vad = SileroVAD(sample_rate=sample_rate) if enable_vad else None
        self.diarizer = SpeakerDiarizer(hf_token=hf_token) if enable_diarization else None

    def process(
        self,
        audio: np.ndarray,
        start_offset: float = 0.0,
    ) -> EnhancedASRResult:
        """
        Process audio through all enhancers.

        Args:
            audio: PCM audio data (int16 or float32)
            start_offset: Time offset for timestamps

        Returns:
            EnhancedASRResult with all enhancement data
        """
        result = EnhancedASRResult()

        # Convert int16 to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # VAD
        if self.vad:
            start = time.time()
            result.vad_segments = self.vad.detect(audio)
            result.vad_time_ms = (time.time() - start) * 1000

            # Calculate speech ratio
            total_speech = sum(s.end - s.start for s in result.vad_segments)
            total_duration = len(audio) / self.sample_rate
            result.speech_ratio = total_speech / total_duration if total_duration > 0 else 0
            result.is_speech = result.speech_ratio > 0.3

        # Diarization (only if speech detected)
        if self.diarizer and result.is_speech:
            start = time.time()
            result.speaker_segments = self.diarizer.diarize(audio, start_offset)
            result.diarization_time_ms = (time.time() - start) * 1000

            # Get current speaker (last segment)
            if result.speaker_segments:
                result.current_speaker = result.speaker_segments[-1].speaker

            # Get all speakers
            result.speakers = self.diarizer.get_speakers()

        return result

    def reset(self):
        """Reset all enhancers for new session."""
        if self.vad:
            self.vad.reset()
        if self.diarizer:
            self.diarizer.reset()


# =============================================================================
# Preload function for startup
# =============================================================================

def preload_models(
    enable_vad: bool = True,
    enable_diarization: bool = False,
    hf_token: Optional[str] = None,
):
    """
    Preload models at startup to avoid first-request delay.

    Note: Diarization requires HF token and is ~2GB, so disabled by default.
    """
    if enable_vad:
        logger.info("Preloading Silero VAD...")
        vad = get_silero_vad()
        if vad:
            logger.info("Silero VAD loaded successfully")
        else:
            logger.warning("Silero VAD failed to load")

    if enable_diarization:
        logger.info("Preloading pyannote diarization (this may take a while)...")
        pipeline = get_diarization_pipeline(hf_token)
        if pipeline:
            logger.info("Pyannote diarization loaded successfully")
        else:
            logger.warning("Pyannote diarization failed to load - check HF_TOKEN")
