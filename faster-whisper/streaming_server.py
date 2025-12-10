#!/usr/bin/env python3
"""
Faster-Whisper Streaming ASR Server

A WebSocket-based streaming server for faster-whisper with:
- Silero VAD for voice activity detection
- Pyannote for speaker diarization
- Built-in punctuation and capitalization
- Multi-language support (EN, FR, DE + 96 more)

GPU Memory: ~4-5GB total (vs 18GB for Voxtral)
License: MIT (commercial use allowed)
"""

import asyncio
import logging
import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Faster-Whisper Streaming ASR",
    description="Multilingual streaming transcription (EN/FR/DE) with VAD and diarization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Configuration
# =============================================================================

# Model Configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # int8 for lower memory

# Audio Configuration
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_DURATION_SECONDS = float(os.environ.get("CHUNK_DURATION", "5.0"))  # Process every 5 seconds (prevents hallucinations)
CHUNK_SIZE_BYTES = int(CHUNK_DURATION_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)

# VAD Configuration
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION = 0.3
SILENCE_DURATION_THRESHOLD = 1.0

# Diarization Configuration
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "true").lower() == "true"
HF_TOKEN = os.environ.get("HF_TOKEN")
MIN_SPEAKERS = 1
MAX_SPEAKERS = 6

# Entity Recognition Configuration
ENABLE_NER = os.environ.get("ENABLE_NER", "true").lower() == "true"

# Smart Formatting Configuration
ENABLE_SMART_FORMAT = os.environ.get("ENABLE_SMART_FORMAT", "true").lower() == "true"

# Hotwords Configuration - Custom vocabulary for better recognition
# Format: comma-separated list of words/phrases
HOTWORDS = os.environ.get("HOTWORDS", "Dudoxx,Walid,Boudabbous,Walid Boudabbous").split(",")
HOTWORDS = [w.strip() for w in HOTWORDS if w.strip()]

# Vocabulary Correction - Map common misrecognitions to correct forms
# Format: JSON string {"misrecognition": "correct", ...}
# These are applied AFTER transcription to fix known errors
VOCAB_CORRECTIONS_RAW = os.environ.get("VOCAB_CORRECTIONS", "{}")
try:
    import json
    VOCAB_CORRECTIONS: dict[str, str] = json.loads(VOCAB_CORRECTIONS_RAW)
except json.JSONDecodeError:
    VOCAB_CORRECTIONS = {}

# Default corrections for Dudoxx (common Whisper misrecognitions)
DEFAULT_CORRECTIONS = {
    # Dudoxx variations
    "didak": "Dudoxx",
    "didaks": "Dudoxx's",
    "didak's": "Dudoxx's",
    "dudok": "Dudoxx",
    "dudoks": "Dudoxx's",
    "dudok's": "Dudoxx's",
    "due dogs": "Dudoxx",
    "due dog": "Dudoxx",
    "do docks": "Dudoxx",
    "do dock": "Dudoxx",
    "dudox": "Dudoxx",
    "dew docks": "Dudoxx",
    "dew dock": "Dudoxx",
    "do dox": "Dudoxx",
    "doodox": "Dudoxx",
    "dodo x": "Dudoxx",
    "du dox": "Dudoxx",
    "du dogs": "Dudoxx",
    "two dogs": "Dudoxx",
    "tu docks": "Dudoxx",
    "g-docs": "Dudoxx",
    "g-doc": "Dudoxx",
    "gdocs": "Dudoxx",
    "d-dox": "Dudoxx",
    "ddox": "Dudoxx",
    "d dox": "Dudoxx",
    "dee dox": "Dudoxx",
    # Medical context corrections (passion→patient is common Whisper error)
    "passion": "patient",
    "passions": "patients",
    "my passion": "my patient",
    "the passion": "the patient",
    "a passion": "a patient",
}
# Merge with env config (env takes precedence)
for k, v in DEFAULT_CORRECTIONS.items():
    if k not in VOCAB_CORRECTIONS:
        VOCAB_CORRECTIONS[k] = v

# Concurrency
MAX_CONCURRENT_REQUESTS = 8

# Request semaphore
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Supported languages
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ar": "Arabic",
}

# =============================================================================
# Global Model Instances (lazy-loaded)
# =============================================================================

_whisper_model = None
_vad_model = None
_diarization_pipeline = None
_ner_model = None
_smart_format_model = None


def get_whisper_model():
    """Lazy-load faster-whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model: {WHISPER_MODEL} ({WHISPER_COMPUTE_TYPE})")
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info(f"faster-whisper model loaded on {WHISPER_DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper: {e}")
            _whisper_model = False
    return _whisper_model if _whisper_model else None


def get_vad_model():
    """Lazy-load Silero VAD model."""
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad(onnx=True)
            logger.info("Silero VAD model loaded (ONNX)")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            _vad_model = False
    return _vad_model if _vad_model else None


def get_diarization_pipeline():
    """Lazy-load pyannote diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None and ENABLE_DIARIZATION and HF_TOKEN:
        try:
            from pyannote.audio import Pipeline
            import torch

            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN,
            )
            if torch.cuda.is_available():
                _diarization_pipeline.to(torch.device("cuda"))
            logger.info("Pyannote diarization pipeline loaded")
        except Exception as e:
            logger.warning(f"Failed to load diarization: {e}")
            _diarization_pipeline = False
    return _diarization_pipeline if _diarization_pipeline else None


def get_ner_model():
    """Lazy-load GLiNER NER model for entity recognition."""
    global _ner_model
    if _ner_model is None and ENABLE_NER:
        try:
            from gliner import GLiNER
            # Use multilingual model for EN/FR/DE support
            _ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
            logger.info("GLiNER NER model loaded (multilingual)")
        except ImportError:
            logger.warning("GLiNER not installed. Run: pip install gliner")
            _ner_model = False
        except Exception as e:
            logger.warning(f"Failed to load GLiNER: {e}")
            _ner_model = False
    return _ner_model if _ner_model else None


def get_smart_format_model():
    """Lazy-load small LLM for smart formatting."""
    global _smart_format_model
    if _smart_format_model is None and ENABLE_SMART_FORMAT:
        try:
            from transformers import pipeline
            import torch

            # Use a small, fast model for formatting
            # Qwen2.5-0.5B is excellent for this task
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _smart_format_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",  # 80MB, very fast
                device=device,
                max_length=512,
            )
            logger.info("Smart formatting model loaded (flan-t5-small)")
        except ImportError:
            logger.warning("Transformers not installed for smart formatting")
            _smart_format_model = False
        except Exception as e:
            logger.warning(f"Failed to load smart format model: {e}")
            _smart_format_model = False
    return _smart_format_model if _smart_format_model else None


# =============================================================================
# Entity Recognition Helper
# =============================================================================

# Entity labels for GLiNER (descriptive labels work better)
NER_LABELS = [
    "person name",
    "company name",
    "organization",
    "city",
    "country",
    "date",
    "time",
    "money amount",
    "phone number",
    "email address",
    "medical term",
    "medication",
]


class EntityRecognizer:
    """Process text with GLiNER for named entity recognition."""

    def __init__(self):
        self.model = get_ner_model()
        self.threshold = 0.3  # Lower threshold for better recall

    def extract_entities(self, text: str, language: str = "en") -> list[dict]:
        """Extract entities from text."""
        if self.model is None or not text.strip():
            return []

        try:
            entities = self.model.predict_entities(
                text,
                NER_LABELS,
                threshold=self.threshold,
            )

            result = []
            for ent in entities:
                result.append({
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": round(ent["score"], 3),
                    "start": ent["start"],
                    "end": ent["end"],
                })

            return result

        except Exception as e:
            logger.error(f"NER error: {e}")
            return []


# =============================================================================
# Fuzzy Hotword Matching
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio (0-1) between two strings."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (distance / max_len)


def apply_fuzzy_hotword_corrections(text: str, threshold: float = 0.7) -> tuple[str, list[dict]]:
    """
    Apply fuzzy matching to correct words similar to hotwords.
    Uses Levenshtein distance with configurable similarity threshold.
    """
    if not text or not HOTWORDS:
        return text, []

    corrections_made = []
    words = text.split()
    result_words = []

    for word in words:
        # Strip punctuation for matching but preserve it
        clean_word = word.strip(".,!?;:'\"()-")
        prefix = word[:len(word) - len(word.lstrip(".,!?;:'\"()-"))]
        suffix = word[len(clean_word) + len(prefix):]

        best_match = None
        best_similarity = threshold

        for hotword in HOTWORDS:
            # For multi-word hotwords, skip single word matching
            if " " in hotword:
                continue

            sim = similarity_ratio(clean_word, hotword)
            if sim > best_similarity and sim < 1.0:  # Not exact match (handled by vocab corrections)
                best_match = hotword
                best_similarity = sim

        if best_match:
            # Preserve original casing style if possible
            if clean_word.isupper():
                corrected = best_match.upper()
            elif clean_word.islower():
                corrected = best_match.lower()
            elif clean_word[0].isupper():
                corrected = best_match.capitalize()
            else:
                corrected = best_match

            corrections_made.append({
                "original": clean_word,
                "corrected": corrected,
                "similarity": round(best_similarity, 2),
                "type": "fuzzy_hotword",
            })
            result_words.append(prefix + corrected + suffix)
        else:
            result_words.append(word)

    return " ".join(result_words), corrections_made


# =============================================================================
# Vocabulary Correction Helper
# =============================================================================

def apply_vocab_corrections(text: str) -> tuple[str, list[dict]]:
    """
    Apply vocabulary corrections to fix common misrecognitions.
    Returns (corrected_text, list_of_corrections_made).
    Case-insensitive matching, preserves original casing style.
    """
    if not text or not VOCAB_CORRECTIONS:
        return text, []

    corrections_made = []
    result = text

    # Sort by length (longer phrases first) to avoid partial replacements
    sorted_corrections = sorted(VOCAB_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)

    for wrong, correct in sorted_corrections:
        # Case-insensitive search
        lower_result = result.lower()
        lower_wrong = wrong.lower()

        start = 0
        while True:
            idx = lower_result.find(lower_wrong, start)
            if idx == -1:
                break

            # Check word boundaries (don't replace partial words)
            before_ok = idx == 0 or not lower_result[idx - 1].isalnum()
            after_idx = idx + len(wrong)
            after_ok = after_idx >= len(lower_result) or not lower_result[after_idx].isalnum()

            if before_ok and after_ok:
                # Get the original text that was matched
                original = result[idx:idx + len(wrong)]

                # Apply correction
                result = result[:idx] + correct + result[idx + len(wrong):]
                lower_result = result.lower()

                corrections_made.append({
                    "original": original,
                    "corrected": correct,
                    "position": idx,
                })

                # Move past this correction
                start = idx + len(correct)
            else:
                start = idx + 1

    return result, corrections_made


# =============================================================================
# Smart Formatting Helper
# =============================================================================

class SmartFormatter:
    """AI-based smart formatting for transcriptions."""

    def __init__(self):
        self.model = get_smart_format_model()

    def format_text(self, text: str, language: str = "en") -> str:
        """Apply smart formatting to text."""
        if self.model is None or not text.strip():
            return text

        try:
            # Prompt for formatting
            lang_name = {"en": "English", "fr": "French", "de": "German"}.get(language, "English")
            prompt = f"""Format this {lang_name} transcription properly:
- Convert spoken numbers to digits (twenty three -> 23)
- Format dates properly (march fifth -> March 5th)
- Format times properly (three thirty pm -> 3:30 PM)
- Format currency (fifty dollars -> $50)
- Keep proper nouns capitalized
- Do not change the meaning

Text: {text}
Formatted:"""

            result = self.model(prompt, max_length=len(text) + 100, do_sample=False)
            formatted = result[0]["generated_text"].strip()

            # Sanity check - don't return if too different
            if len(formatted) > len(text) * 2 or len(formatted) < len(text) * 0.3:
                return text

            return formatted

        except Exception as e:
            logger.error(f"Smart format error: {e}")
            return text


# =============================================================================
# Preload Models at Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    """Preload models at startup."""
    logger.info("Preloading models...")
    logger.info(f"Hotwords configured: {HOTWORDS}")
    logger.info(f"Vocabulary corrections: {len(VOCAB_CORRECTIONS)} patterns loaded")

    # Load Whisper
    model = get_whisper_model()
    if model:
        logger.info("faster-whisper ready")

    # Load VAD
    vad = get_vad_model()
    if vad:
        logger.info("Silero VAD ready")

    # Load diarization
    if ENABLE_DIARIZATION and HF_TOKEN:
        diarize = get_diarization_pipeline()
        if diarize:
            logger.info("Pyannote diarization ready")
    elif ENABLE_DIARIZATION:
        logger.warning("Diarization enabled but HF_TOKEN not set")

    # Load NER model
    if ENABLE_NER:
        ner = get_ner_model()
        if ner:
            logger.info("GLiNER NER ready")

    # Load smart formatting model
    if ENABLE_SMART_FORMAT:
        fmt = get_smart_format_model()
        if fmt:
            logger.info("Smart formatting ready")

    logger.info("All models loaded")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")


# =============================================================================
# VAD Helper
# =============================================================================

@dataclass
class VADSegment:
    """Voice activity segment."""
    start: float
    end: float


class SileroVADProcessor:
    """Process audio with Silero VAD."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = get_vad_model()

    def detect(self, audio: np.ndarray) -> tuple[list[VADSegment], float]:
        """
        Detect speech segments.

        Returns:
            (segments, speech_ratio)
        """
        if self.model is None:
            return [], 1.0

        try:
            import torch
            from silero_vad import get_speech_timestamps

            # Ensure float32 normalized
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            audio_tensor = torch.from_numpy(audio)

            timestamps = get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )

            segments = []
            for ts in timestamps:
                segments.append(VADSegment(
                    start=ts["start"] / self.sample_rate,
                    end=ts["end"] / self.sample_rate,
                ))

            # Calculate speech ratio
            total_speech = sum(s.end - s.start for s in segments)
            total_duration = len(audio) / self.sample_rate
            speech_ratio = total_speech / total_duration if total_duration > 0 else 0

            return segments, speech_ratio

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return [], 1.0

    def is_speech(self, audio: np.ndarray, min_ratio: float = 0.3) -> bool:
        """Check if audio contains sufficient speech."""
        _, ratio = self.detect(audio)
        return ratio >= min_ratio


# =============================================================================
# Speaker Diarization Helper
# =============================================================================

@dataclass
class SpeakerSegment:
    """Speaker segment."""
    speaker: str
    start: float
    end: float


class DiarizationProcessor:
    """Process audio with pyannote diarization."""

    def __init__(self):
        self.pipeline = get_diarization_pipeline()
        self.speakers: dict[str, str] = {}  # raw_id -> label
        self.speaker_count = 0

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeakerSegment]:
        """Perform speaker diarization."""
        if self.pipeline is None:
            return []

        try:
            import torch

            # Prepare audio
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            waveform = torch.from_numpy(audio).unsqueeze(0)
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

            # Run diarization
            result = self.pipeline(
                audio_dict,
                min_speakers=MIN_SPEAKERS,
                max_speakers=MAX_SPEAKERS,
            )

            # Pyannote 4.x returns DiarizeOutput with speaker_diarization attribute
            # (which is an Annotation object with itertracks method)
            diarization = getattr(result, 'speaker_diarization', result)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Track speaker labels
                if speaker not in self.speakers:
                    self.speaker_count += 1
                    self.speakers[speaker] = f"Speaker {self.speaker_count}"

                segments.append(SpeakerSegment(
                    speaker=self.speakers[speaker],
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                ))

            return segments

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            return []

    def get_current_speaker(self, segments: list[SpeakerSegment]) -> Optional[str]:
        """Get the current (last) speaker."""
        return segments[-1].speaker if segments else None

    def get_speakers_list(self) -> list[dict]:
        """Get list of all detected speakers."""
        return [{"id": k, "label": v} for k, v in self.speakers.items()]

    def reset(self):
        """Reset for new session."""
        self.speakers.clear()
        self.speaker_count = 0


# =============================================================================
# Streaming Processor
# =============================================================================

@dataclass
class TranscriptionResult:
    """Result from transcription."""
    text: str
    language: str
    language_probability: float
    segments: list[dict] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)


class FasterWhisperStreamingProcessor:
    """Handles chunked audio processing with VAD for streaming transcription."""

    def __init__(self, language: str = "auto"):
        self.language = language if language != "auto" else None
        self.audio_buffer = bytearray()
        self.full_transcript: list[str] = []
        self.last_speech_time = time.time()
        self.speech_detected = False

        # Models
        self.whisper = get_whisper_model()
        self.vad = SileroVADProcessor()
        self.diarizer = DiarizationProcessor() if ENABLE_DIARIZATION else None
        self.ner = EntityRecognizer() if ENABLE_NER else None
        self.formatter = SmartFormatter() if ENABLE_SMART_FORMAT else None

        # Hotwords for custom vocabulary
        self.hotwords = " ".join(HOTWORDS) if HOTWORDS else None

        # All detected entities across session
        self.all_entities: list[dict] = []

        # Statistics
        self.start_time = time.time()
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self.chunk_count = 0
        self.word_count = 0
        self.detected_language = "unknown"

        # Timestamp tracking
        self.current_audio_position = 0.0

    async def process_chunk(self, audio_data: bytes) -> Optional[dict]:
        """Process an audio chunk and return transcription result."""
        self.audio_buffer.extend(audio_data)

        # Check buffer size
        buffer_duration = len(self.audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        # Convert to numpy for VAD check
        if len(self.audio_buffer) >= SAMPLE_RATE * BYTES_PER_SAMPLE * 0.5:  # At least 0.5s
            audio_np = np.frombuffer(bytes(self.audio_buffer[-SAMPLE_RATE * BYTES_PER_SAMPLE:]), dtype=np.int16)
            is_speech = self.vad.is_speech(audio_np, min_ratio=0.2)

            if is_speech:
                self.speech_detected = True
                self.last_speech_time = time.time()

        # Check if we should process
        silence_duration = time.time() - self.last_speech_time
        should_process = False
        is_final = False

        if len(self.audio_buffer) >= CHUNK_SIZE_BYTES and self.speech_detected:
            should_process = True
        elif silence_duration > SILENCE_DURATION_THRESHOLD and self.speech_detected and buffer_duration > MIN_SPEECH_DURATION:
            should_process = True
            is_final = True
            self.speech_detected = False

        if not should_process:
            return None

        # Extract chunk (clean 5-second chunks, no overlap to prevent repetition)
        if is_final:
            chunk = bytes(self.audio_buffer)
            self.audio_buffer.clear()
        else:
            chunk = bytes(self.audio_buffer[:CHUNK_SIZE_BYTES])
            # Remove processed chunk completely (no overlap)
            self.audio_buffer = self.audio_buffer[CHUNK_SIZE_BYTES:]

        # Process chunk
        return await self._transcribe_chunk(chunk, is_final)

    async def _transcribe_chunk(self, chunk: bytes, is_final: bool) -> Optional[dict]:
        """Transcribe a single chunk."""
        if self.whisper is None:
            return None

        chunk_duration = len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        self.total_audio_duration += chunk_duration
        self.chunk_count += 1

        # Convert to numpy float32
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # VAD analysis
        vad_segments, speech_ratio = self.vad.detect(audio_np)

        # Skip if no speech
        if speech_ratio < 0.2:
            return None

        # Transcribe
        proc_start = time.time()

        try:
            # Build transcribe kwargs
            # NOTE: initial_prompt causes hallucinations - removed!
            # We rely on hotwords + post-processing vocab corrections instead
            transcribe_kwargs = dict(
                language=self.language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            # Add hotwords if configured (boosts recognition of custom vocabulary)
            if self.hotwords:
                transcribe_kwargs["hotwords"] = self.hotwords

            segments, info = self.whisper.transcribe(audio_np, **transcribe_kwargs)

            # Collect results
            text_parts = []
            word_list = []
            segment_list = []

            for segment in segments:
                text_parts.append(segment.text.strip())

                segment_list.append({
                    "start": round(segment.start + self.current_audio_position, 3),
                    "end": round(segment.end + self.current_audio_position, 3),
                    "text": segment.text.strip(),
                })

                if segment.words:
                    for word in segment.words:
                        word_list.append({
                            "word": word.word.strip(),
                            "start": round(word.start + self.current_audio_position, 3),
                            "end": round(word.end + self.current_audio_position, 3),
                            "probability": round(word.probability, 3),
                        })

            text = " ".join(text_parts).strip()

            if not text:
                return None

            # Apply vocabulary corrections (fix common misrecognitions)
            text, vocab_corrections = apply_vocab_corrections(text)

            # Apply fuzzy hotword corrections (catch similar-sounding words)
            text, fuzzy_corrections = apply_fuzzy_hotword_corrections(text)
            vocab_corrections.extend(fuzzy_corrections)

            # Also update segment texts with corrections
            for seg in segment_list:
                seg["text"], _ = apply_vocab_corrections(seg["text"])
                seg["text"], _ = apply_fuzzy_hotword_corrections(seg["text"])

            proc_time = time.time() - proc_start
            self.total_processing_time += proc_time

            # Update state
            self.full_transcript.append(text)
            self.word_count += len(text.split())
            self.detected_language = info.language

            # Diarization
            diarization_data = None
            if self.diarizer:
                diarize_start = time.time()
                speaker_segments = self.diarizer.diarize(audio_np, SAMPLE_RATE)
                diarize_time = (time.time() - diarize_start) * 1000

                if speaker_segments:
                    # Adjust timestamps
                    for seg in speaker_segments:
                        seg.start += self.current_audio_position
                        seg.end += self.current_audio_position

                    diarization_data = {
                        "current_speaker": self.diarizer.get_current_speaker(speaker_segments),
                        "segments": [
                            {"speaker": s.speaker, "start": s.start, "end": s.end}
                            for s in speaker_segments
                        ],
                        "speakers": self.diarizer.get_speakers_list(),
                        "processing_ms": round(diarize_time, 1),
                    }

            # Entity Recognition (NER)
            entities_data = None
            if self.ner:
                ner_start = time.time()
                chunk_entities = self.ner.extract_entities(text, info.language)
                ner_time = (time.time() - ner_start) * 1000

                if chunk_entities:
                    # Track all entities across session
                    for ent in chunk_entities:
                        # Add to session entities if not duplicate
                        if not any(e["text"] == ent["text"] and e["label"] == ent["label"]
                                   for e in self.all_entities):
                            self.all_entities.append(ent)

                    entities_data = {
                        "chunk_entities": chunk_entities,
                        "session_entities": self.all_entities,
                        "processing_ms": round(ner_time, 1),
                    }

            # Smart Formatting
            formatted_text = text
            formatting_data = None
            if self.formatter and is_final:  # Only format final chunks to save processing
                format_start = time.time()
                formatted_text = self.formatter.format_text(text, info.language)
                format_time = (time.time() - format_start) * 1000

                if formatted_text != text:
                    formatting_data = {
                        "original": text,
                        "formatted": formatted_text,
                        "processing_ms": round(format_time, 1),
                    }
                    # Update full transcript with formatted text
                    if self.full_transcript:
                        self.full_transcript[-1] = formatted_text

            # Calculate timestamps
            chunk_start = self.current_audio_position
            chunk_end = chunk_start + chunk_duration
            self.current_audio_position = chunk_end

            # Calculate metrics
            rtf = proc_time / chunk_duration if chunk_duration > 0 else 0
            session_duration = time.time() - self.start_time
            wpm = (self.word_count / session_duration * 60) if session_duration > 0 else 0

            return {
                "type": "final" if is_final else "partial",
                "text": formatted_text if formatting_data else text,
                "full_transcript": self.get_full_transcript(),
                "timing": {
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "duration": round(chunk_duration, 3),
                },
                "segments": segment_list,
                "words": word_list,
                "language": {
                    "detected": info.language,
                    "probability": round(info.language_probability, 3),
                    "requested": self.language or "auto",
                },
                "diarization": diarization_data,
                "entities": entities_data,
                "formatting": formatting_data,
                "vad": {
                    "is_speech": True,
                    "speech_ratio": round(speech_ratio, 2),
                    "segments": [{"start": s.start, "end": s.end} for s in vad_segments],
                },
                "processing": {
                    "latency_ms": round(proc_time * 1000, 1),
                    "rtf": round(rtf, 3),
                    "chunk_index": self.chunk_count,
                    "hotwords_enabled": bool(self.hotwords),
                    "ner_enabled": self.ner is not None,
                    "formatting_enabled": self.formatter is not None,
                    "vocab_corrections": vocab_corrections if vocab_corrections else None,
                },
                "stats": {
                    "word_count": self.word_count,
                    "wpm": round(wpm, 1),
                    "total_audio_sec": round(self.total_audio_duration, 2),
                    "total_proc_ms": round(self.total_processing_time * 1000, 1),
                    "total_speakers": len(self.diarizer.speakers) if self.diarizer else 0,
                    "total_entities": len(self.all_entities),
                },
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def flush(self) -> Optional[dict]:
        """Process remaining audio."""
        if len(self.audio_buffer) < SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_SPEECH_DURATION:
            return None

        chunk = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        return await self._transcribe_chunk(chunk, is_final=True)

    def get_full_transcript(self) -> str:
        """Get complete transcript."""
        return " ".join(self.full_transcript)


# =============================================================================
# HTML Frontend
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Faster-Whisper Streaming ASR</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
               margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .panel { background: #16213e; border-radius: 12px; padding: 20px; }
        h1 { color: #00d4ff; margin: 0 0 5px 0; font-size: 24px; }
        h2 { color: #00d4ff; margin: 0 0 15px 0; font-size: 16px; border-bottom: 1px solid #0f3460; padding-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 15px; font-size: 13px; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px; }
        .badge-green { background: #1b4332; color: #95d5b2; }
        .badge-blue { background: #0f3460; color: #00d4ff; }
        .status { padding: 10px 15px; border-radius: 6px; margin: 10px 0; font-weight: 500; font-size: 13px; }
        .connected { background: #1b4332; color: #95d5b2; }
        .disconnected { background: #4a1515; color: #f8d7da; }
        .recording { background: #4a3f15; color: #fff3cd; }
        .controls { display: flex; gap: 8px; margin: 15px 0; flex-wrap: wrap; align-items: center; }
        button { padding: 10px 18px; border: none; border-radius: 6px; cursor: pointer;
                 font-size: 13px; font-weight: 500; transition: all 0.2s; font-family: inherit; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-primary { background: #0077b6; color: white; }
        .btn-primary:hover:not(:disabled) { background: #005f8a; }
        .btn-success { background: #2d6a4f; color: white; }
        .btn-success:hover:not(:disabled) { background: #1b4332; }
        .btn-danger { background: #9d0208; color: white; }
        .btn-danger:hover:not(:disabled) { background: #6a040f; }
        .btn-secondary { background: #4a4e69; color: white; }
        select { padding: 8px 12px; border-radius: 6px; border: 1px solid #0f3460;
                 font-size: 13px; background: #1a1a2e; color: #eee; }
        #transcript { min-height: 200px; max-height: 500px; overflow-y: auto; border: 2px solid #0f3460; border-radius: 8px;
                     padding: 15px; margin: 15px 0; background: #0f3460; font-size: 15px; line-height: 1.6; scroll-behavior: smooth; }
        .partial { color: #ffd166; font-style: italic; animation: pulse 1.5s ease-in-out infinite; }
        .segment-block { margin-bottom: 12px; animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .speaker-label { color: #40916c; font-weight: bold; margin-right: 6px; }
        .speaker-1 { color: #40916c; }
        .speaker-2 { color: #f77f00; }
        .speaker-3 { color: #9d4edd; }
        .speaker-4 { color: #00d4ff; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }
        .stat-box { background: #0f3460; padding: 10px; border-radius: 6px; text-align: center; }
        .stat-label { font-size: 11px; color: #888; margin-bottom: 4px; }
        .stat-value { font-size: 18px; color: #00d4ff; font-weight: bold; }
        .energy-bar { height: 8px; background: #0f3460; border-radius: 4px; overflow: hidden; margin-top: 5px; }
        .energy-level { height: 100%; background: linear-gradient(90deg, #2d6a4f, #40916c, #ffd166, #f77f00); transition: width 0.1s; }

        .debug-panel { height: calc(100vh - 40px); display: flex; flex-direction: column; }
        .debug-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .debug-controls { display: flex; gap: 8px; }
        #debugLog { flex: 1; background: #0a0a14; border: 1px solid #0f3460; border-radius: 8px;
                   padding: 10px; overflow-y: auto; font-family: 'Fira Code', 'Consolas', monospace;
                   font-size: 11px; line-height: 1.5; }
        .log-entry { padding: 4px 0; border-bottom: 1px solid #1a1a2e; }
        .log-time { color: #666; margin-right: 8px; }
        .log-send { color: #f77f00; }
        .log-recv { color: #2ec4b6; }
        .log-info { color: #888; }
        .log-error { color: #ef476f; }
        .log-data { color: #ffd166; margin-left: 20px; display: block; white-space: pre-wrap; word-break: break-all; max-height: 150px; overflow-y: auto; }

        .protocol-info { background: #0f3460; border-radius: 8px; padding: 15px; margin-top: 15px; font-size: 12px; }
        .protocol-info h3 { color: #00d4ff; margin: 0 0 10px 0; font-size: 14px; }
        .protocol-info code { background: #1a1a2e; padding: 2px 6px; border-radius: 3px; color: #ffd166; }

        .entity-tag { display: inline-block; padding: 2px 6px; border-radius: 3px; margin: 2px; font-size: 11px; }
        .entity-person { background: #1b4332; color: #95d5b2; }
        .entity-org { background: #0f3460; color: #00d4ff; }
        .entity-location { background: #4a3f15; color: #ffd166; }
        .entity-date { background: #4a1515; color: #f8d7da; }
        .entity-other { background: #2d2d44; color: #aaa; }
        .correction-item { padding: 3px 0; border-bottom: 1px solid #1a1a2e; }
        .correction-original { color: #f77f00; text-decoration: line-through; }
        .correction-arrow { color: #666; margin: 0 6px; }
        .correction-fixed { color: #40916c; font-weight: bold; }
        .transcript-entity { padding: 1px 4px; border-radius: 3px; }
        .transcript-entity-person { background: rgba(27, 67, 50, 0.5); border-bottom: 2px solid #40916c; }
        .transcript-entity-org { background: rgba(15, 52, 96, 0.5); border-bottom: 2px solid #00d4ff; }
        .transcript-entity-location { background: rgba(74, 63, 21, 0.5); border-bottom: 2px solid #ffd166; }

        @media (max-width: 1000px) {
            .layout { grid-template-columns: 1fr; }
            .debug-panel { height: 500px; }
        }
    </style>
</head>
<body>
    <div class="layout">
        <div class="panel">
            <h1>Faster-Whisper Streaming ASR
                <span class="badge badge-green">~4GB GPU</span>
                <span class="badge badge-blue">MIT License</span>
            </h1>
            <p class="subtitle">Multilingual streaming ASR (EN/FR/DE) with VAD + Diarization | INT8 Quantized</p>

            <div id="status" class="status disconnected">Disconnected</div>

            <div class="controls">
                <select id="language">
                    <option value="auto">Auto-detect</option>
                    <option value="en">English</option>
                    <option value="de">German</option>
                    <option value="fr">French</option>
                    <option value="es">Spanish</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                    <option value="nl">Dutch</option>
                    <option value="ar">Arabic</option>
                </select>
                <button id="connectBtn" class="btn-primary" onclick="connect()">Connect</button>
                <button id="startBtn" class="btn-success" onclick="startRecording()" disabled>Start Recording</button>
                <button id="stopBtn" class="btn-danger" onclick="stopRecording()" disabled>Stop Recording</button>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">LANGUAGE</div>
                    <div class="stat-value" id="langVal">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">CONFIDENCE</div>
                    <div class="stat-value" id="langConf">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">LATENCY (ms)</div>
                    <div class="stat-value" id="latency">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">RTF</div>
                    <div class="stat-value" id="rtfVal">-</div>
                </div>
            </div>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">WORDS</div>
                    <div class="stat-value" id="wordCount">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">WPM</div>
                    <div class="stat-value" id="wpmVal">-</div>
                </div>
                <div class="stat-box" style="background: #1b4332;">
                    <div class="stat-label">SPEAKER</div>
                    <div class="stat-value" id="currentSpeaker">-</div>
                </div>
                <div class="stat-box" style="background: #1b4332;">
                    <div class="stat-label">SPEAKERS</div>
                    <div class="stat-value" id="speakerCount">0</div>
                </div>
            </div>
            <div class="stats" style="grid-template-columns: repeat(3, 1fr);">
                <div class="stat-box">
                    <div class="stat-label">VAD</div>
                    <div class="stat-value" id="vadStatus">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">SPEECH %</div>
                    <div class="stat-value" id="speechRatio">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">AUDIO TIME</div>
                    <div class="stat-value" id="audioTime">0s</div>
                </div>
            </div>

            <h2>Transcript</h2>
            <div id="transcript"></div>

            <div class="stats" style="grid-template-columns: 1fr 1fr; margin-top: 15px;">
                <div class="stat-box" style="text-align: left;">
                    <div class="stat-label">DETECTED ENTITIES</div>
                    <div id="entitiesContainer" style="font-size: 12px; margin-top: 8px; max-height: 100px; overflow-y: auto;"></div>
                </div>
                <div class="stat-box" style="text-align: left;">
                    <div class="stat-label">CORRECTIONS APPLIED</div>
                    <div id="correctionsContainer" style="font-size: 12px; margin-top: 8px; max-height: 100px; overflow-y: auto;"></div>
                </div>
            </div>

            <div style="margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h2 style="margin:0; border:none; padding:0;">Renderer Log (Client-Side)</h2>
                    <button class="btn-secondary" onclick="clearRendererLog()" style="padding: 4px 10px; font-size: 11px;">Clear</button>
                </div>
                <div id="rendererLog" style="background: #0a0a14; border: 1px solid #0f3460; border-radius: 8px; padding: 10px; max-height: 200px; overflow-y: auto; font-family: 'Fira Code', 'Consolas', monospace; font-size: 11px; line-height: 1.5;"></div>
            </div>

            <div class="protocol-info">
                <h3>WebSocket Protocol</h3>
                <p><strong>Endpoint:</strong> <code>wss://canary.dudoxx.com/asr?language=auto</code></p>
                <p><strong>Audio Format:</strong> PCM Int16, 16kHz, Mono</p>
                <p><strong>Model:</strong> faster-whisper large-v3 (INT8)</p>
                <p><strong>Features:</strong> VAD, Diarization, Word timestamps, Auto punctuation</p>
            </div>
        </div>

        <div class="panel debug-panel">
            <div class="debug-header">
                <h2 style="margin:0; border:none; padding:0;">Debug Log</h2>
                <div class="debug-controls">
                    <button class="btn-secondary" onclick="clearLog()">Clear</button>
                    <button class="btn-secondary" onclick="copyLog()">Copy</button>
                    <label style="font-size:12px; color:#888;">
                        <input type="checkbox" id="autoScroll" checked> Auto-scroll
                    </label>
                </div>
            </div>
            <div id="debugLog"></div>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let processor = null;
        let isRecording = false;
        let chunkCount = 0;
        let bytesSent = 0;
        let transcriptSegments = [];  // Store all final segments with speaker and timestamp
        let lastSpeaker = null;
        let allCorrections = [];
        let cumulativeCorrections = [];  // Track all corrections made

        function formatTimestamp(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins + ':' + (secs < 10 ? '0' : '') + secs;
        }

        function rendererLog(message, data) {
            const rendererLogDiv = document.getElementById('rendererLog');
            const time = new Date().toISOString().substr(11, 12);
            const entry = document.createElement('div');
            entry.style.padding = '3px 0';
            entry.style.borderBottom = '1px solid #1a1a2e';

            let content = '<span style="color: #666;">' + time + '</span> ';
            content += '<span style="color: #2ec4b6;">' + message + '</span>';
            if (data !== undefined) {
                content += '<br><span style="color: #ffd166; margin-left: 10px; font-size: 10px;">' + JSON.stringify(data) + '</span>';
            }
            entry.innerHTML = content;

            rendererLogDiv.appendChild(entry);
            rendererLogDiv.scrollTop = rendererLogDiv.scrollHeight;
        }

        function clearRendererLog() {
            document.getElementById('rendererLog').innerHTML = '';
            rendererLog('Renderer log cleared');
        }

        function log(type, direction, data) {
            const debugLog = document.getElementById('debugLog');
            const time = new Date().toISOString().substr(11, 12);
            const entry = document.createElement('div');
            entry.className = 'log-entry';

            let dirClass = 'log-info';
            let dirLabel = '[INFO]';
            if (direction === 'send') { dirClass = 'log-send'; dirLabel = '[SEND]'; }
            else if (direction === 'recv') { dirClass = 'log-recv'; dirLabel = '[RECV]'; }
            else if (direction === 'error') { dirClass = 'log-error'; dirLabel = '[ERROR]'; }

            let dataStr = '';
            if (data !== undefined) {
                if (typeof data === 'object') {
                    dataStr = '<span class="log-data">' + JSON.stringify(data, null, 2) + '</span>';
                } else {
                    dataStr = '<span class="log-data">' + data + '</span>';
                }
            }

            entry.innerHTML = '<span class="log-time">' + time + '</span><span class="' + dirClass + '">' + dirLabel + '</span> ' + type + dataStr;
            debugLog.appendChild(entry);

            if (document.getElementById('autoScroll').checked) {
                debugLog.scrollTop = debugLog.scrollHeight;
            }
        }

        function clearLog() { document.getElementById('debugLog').innerHTML = ''; }
        function copyLog() { navigator.clipboard.writeText(document.getElementById('debugLog').innerText); }

        function updateStatus(status) {
            const el = document.getElementById('status');
            el.className = 'status ' + status;
            const texts = { connected: 'Connected', disconnected: 'Disconnected', recording: 'Recording...' };
            el.textContent = texts[status] || status;
        }

        function connect() {
            const lang = document.getElementById('language').value;
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + location.host + '/asr?language=' + lang;

            log('Connecting', 'info', wsUrl);
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                log('Connected', 'recv');
                updateStatus('connected');
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('startBtn').disabled = false;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    log('Message', 'recv', data);

                    if (data.type === 'partial' || data.type === 'final') {
                        const speaker = data.diarization?.current_speaker || 'Speaker 1';

                        rendererLog('📥 Received ' + data.type, {
                            text: data.text,
                            speaker: speaker,
                            timestamp: data.timing?.start
                        });

                        // Add final segments to history
                        if (data.type === 'final' && data.segments) {
                            rendererLog('💾 Storing ' + data.segments.length + ' final segment(s)', {
                                before: transcriptSegments.length,
                                adding: data.segments.length
                            });

                            data.segments.forEach(seg => {
                                transcriptSegments.push({
                                    text: seg.text,
                                    speaker: speaker,
                                    timestamp: seg.start
                                });
                            });

                            rendererLog('✓ Total segments: ' + transcriptSegments.length);
                        }

                        // Build HTML from segments
                        rendererLog('🎨 Building HTML...', {segments: transcriptSegments.length});
                        let html = '';

                        // Render all final segments
                        transcriptSegments.forEach((seg, idx) => {
                            const speakerNum = seg.speaker.replace('Speaker ', '');
                            const speakerClass = 'speaker-' + Math.min(parseInt(speakerNum) || 1, 4);
                            const timeStr = formatTimestamp(seg.timestamp);

                            html += '<div class="segment-block">';
                            html += '<strong class="speaker-label ' + speakerClass + '">' + seg.speaker + '</strong><br>';
                            html += '<span style="color: #666; font-size: 11px;">' + timeStr + '</span><br>';
                            html += '<span>' + seg.text + '</span>';
                            html += '</div>';
                        });

                        // Add partial text if available
                        if (data.type === 'partial' && data.text) {
                            const speakerNum = speaker.replace('Speaker ', '');
                            const speakerClass = 'speaker-' + Math.min(parseInt(speakerNum) || 1, 4);
                            const currentTime = data.timing?.start || 0;
                            const timeStr = formatTimestamp(currentTime);

                            html += '<div class="segment-block">';
                            html += '<strong class="speaker-label ' + speakerClass + '">' + speaker + '</strong><br>';
                            html += '<span style="color: #666; font-size: 11px;">' + timeStr + '</span><br>';
                            html += '<span class="partial">Listening...<br>' + data.text + '</span>';
                            html += '</div>';
                        }

                        rendererLog('📝 Rendering to DOM', {
                            html_length: html.length,
                            has_partial: data.type === 'partial'
                        });

                        const transcriptDiv = document.getElementById('transcript');
                        transcriptDiv.innerHTML = html;

                        // Auto-scroll to bottom when new content appears
                        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;

                        rendererLog('✅ Rendered successfully', {
                            scroll_height: transcriptDiv.scrollHeight,
                            visible_segments: transcriptSegments.length
                        });
                    }

                    // Update stats
                    if (data.language) {
                        document.getElementById('langVal').textContent = (data.language.detected || '-').toUpperCase();
                        document.getElementById('langConf').textContent = data.language.probability ? Math.round(data.language.probability * 100) + '%' : '-';
                    }
                    if (data.processing) {
                        document.getElementById('latency').textContent = data.processing.latency_ms || '-';
                        document.getElementById('rtfVal').textContent = data.processing.rtf || '-';
                    }
                    if (data.stats) {
                        document.getElementById('wordCount').textContent = data.stats.word_count || 0;
                        document.getElementById('wpmVal').textContent = data.stats.wpm || '-';
                        document.getElementById('audioTime').textContent = (data.stats.total_audio_sec || 0).toFixed(1) + 's';
                        document.getElementById('speakerCount').textContent = data.stats.total_speakers || 0;
                    }
                    if (data.diarization) {
                        document.getElementById('currentSpeaker').textContent = data.diarization.current_speaker || '-';
                    }
                    if (data.vad) {
                        document.getElementById('vadStatus').textContent = data.vad.is_speech ? 'SPEECH' : 'SILENT';
                        document.getElementById('speechRatio').textContent = Math.round(data.vad.speech_ratio * 100) + '%';
                    }

                    // Update entities display
                    if (data.entities && data.entities.session_entities) {
                        const entitiesContainer = document.getElementById('entitiesContainer');
                        const uniqueEntities = {};
                        data.entities.session_entities.forEach(e => {
                            const key = e.text + ':' + e.label;
                            if (!uniqueEntities[key] || e.score > uniqueEntities[key].score) {
                                uniqueEntities[key] = e;
                            }
                        });

                        let entitiesHtml = '';
                        Object.values(uniqueEntities).forEach(e => {
                            let cls = 'entity-other';
                            if (e.label.includes('person')) cls = 'entity-person';
                            else if (e.label.includes('org') || e.label.includes('company')) cls = 'entity-org';
                            else if (e.label.includes('city') || e.label.includes('country') || e.label.includes('location')) cls = 'entity-location';
                            else if (e.label.includes('date') || e.label.includes('time')) cls = 'entity-date';
                            entitiesHtml += '<span class="entity-tag ' + cls + '">' + e.text + ' <small>(' + e.label + ')</small></span>';
                        });
                        entitiesContainer.innerHTML = entitiesHtml || '<span style="color:#666">No entities detected</span>';
                    }

                    // Update corrections display (cumulative)
                    if (data.processing && data.processing.vocab_corrections) {
                        // Add new corrections to cumulative list
                        data.processing.vocab_corrections.forEach(c => {
                            // Check if not already in list
                            if (!cumulativeCorrections.some(cc => cc.original === c.original && cc.corrected === c.corrected)) {
                                cumulativeCorrections.push(c);
                            }
                        });

                        const correctionsContainer = document.getElementById('correctionsContainer');
                        let correctionsHtml = '';
                        cumulativeCorrections.forEach(c => {
                            const typeLabel = c.type === 'fuzzy_hotword' ? ' (fuzzy)' : '';
                            correctionsHtml += '<div class="correction-item">' +
                                '<span class="correction-original">' + c.original + '</span>' +
                                '<span class="correction-arrow">→</span>' +
                                '<span class="correction-fixed">' + c.corrected + typeLabel + '</span></div>';
                        });
                        correctionsContainer.innerHTML = correctionsHtml || '<span style="color:#666">No corrections applied</span>';
                    }
                } catch (error) {
                    rendererLog('❌ ERROR: ' + error.message);
                    console.error('Full error:', error);
                }
            };

            ws.onclose = () => {
                log('Disconnected', 'recv');
                updateStatus('disconnected');
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('startBtn').disabled = true;
                stopRecording();
            };

            ws.onerror = (err) => { log('Error', 'error', err.message); };
        }

        async function startRecording() {
            if (isRecording) return;

            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(2048, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcm = new Int16Array(inputData.length);

                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }

                    ws.send(pcm.buffer);
                    chunkCount++;
                    bytesSent += pcm.buffer.byteLength;

                    // Reduced logging - only log every 100th chunk to avoid spam
                    if (chunkCount % 100 === 0) {
                        log('Audio streaming', 'info', { chunks: chunkCount, total_bytes: bytesSent });
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                chunkCount = 0;
                bytesSent = 0;
                lastSpeaker = null;
                transcriptSegments = [];  // Reset transcript history
                cumulativeCorrections = [];  // Reset corrections
                document.getElementById('entitiesContainer').innerHTML = '<span style="color:#666">No entities detected</span>';
                document.getElementById('correctionsContainer').innerHTML = '<span style="color:#666">No corrections applied</span>';
                updateStatus('recording');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;

                rendererLog('🎙️ Recording started', {
                    sampleRate: 16000,
                    channelCount: 1,
                    bufferSize: 2048
                });
                log('Recording started', 'info');

            } catch (err) {
                log('Microphone error', 'error', err.message);
            }
        }

        function stopRecording() {
            if (!isRecording && !processor) return;

            isRecording = false;

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(new ArrayBuffer(0));
            }

            if (processor) processor.disconnect();
            if (audioContext) audioContext.close();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());

            updateStatus('connected');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;

            log('Recording stopped', 'info');
        }

        log('Page loaded', 'info');
        rendererLog('🚀 Renderer initialized', {
            browser: navigator.userAgent.split(' ').pop(),
            timestamp: new Date().toISOString()
        });
    </script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve demo page."""
    return HTML_TEMPLATE


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket, language: str = "auto"):
    """WebSocket endpoint for streaming ASR."""
    await websocket.accept()
    logger.info(f"WebSocket connected, language: {language}")

    processor = FasterWhisperStreamingProcessor(language=language)

    try:
        await websocket.send_json({
            "type": "config",
            "language": language,
            "model": f"faster-whisper {WHISPER_MODEL}",
            "compute_type": WHISPER_COMPUTE_TYPE,
            "vad_enabled": True,
            "diarization_enabled": ENABLE_DIARIZATION,
        })

        while True:
            message = await websocket.receive_bytes()

            if not message:
                result = await processor.flush()
                if result:
                    await websocket.send_json(result)
                break

            result = await processor.process_chunk(message)
            if result:
                await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="auto"),
):
    """Transcribe uploaded audio file."""
    async with request_semaphore:
        try:
            import tempfile
            import os as os_module

            content = await file.read()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(content)
                temp_path = f.name

            try:
                whisper = get_whisper_model()
                if whisper is None:
                    return JSONResponse(status_code=500, content={"error": "Model not loaded"})

                segments, info = whisper.transcribe(
                    temp_path,
                    language=language if language != "auto" else None,
                    beam_size=5,
                    word_timestamps=True,
                )

                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())

                return {
                    "text": " ".join(text_parts),
                    "language": info.language,
                    "language_probability": round(info.language_probability, 3),
                }

            finally:
                os_module.unlink(temp_path)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health():
    """Health check."""
    whisper = get_whisper_model()
    vad = get_vad_model()
    diarize = get_diarization_pipeline()
    ner = get_ner_model()
    formatter = get_smart_format_model()

    return {
        "status": "healthy" if whisper else "degraded",
        "model": f"faster-whisper {WHISPER_MODEL}",
        "compute_type": WHISPER_COMPUTE_TYPE,
        "whisper_loaded": whisper is not None,
        "vad_loaded": vad is not None,
        "diarization_loaded": diarize is not None,
        "ner_loaded": ner is not None,
        "smart_format_loaded": formatter is not None,
        "hotwords": HOTWORDS,
        "languages": list(SUPPORTED_LANGUAGES.keys()),
    }


@app.get("/api/languages")
async def get_languages():
    """Get supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Faster-Whisper Streaming ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=4400, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
