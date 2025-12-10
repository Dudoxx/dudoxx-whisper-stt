#!/usr/bin/env python3
"""
Voxtral Mini 3B Streaming ASR Server - Optimized

A WebSocket-based streaming server for Voxtral Mini 3B transcription.
Supports chunked audio processing with VAD for near-real-time transcription.

License: Apache 2.0 (Commercial use allowed)
"""

import asyncio
import logging
import os
import struct
import time
from collections import deque
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ASR Enhancement modules
from asr_enhancers import ASREnhancer, preload_models, SileroVAD

# Advanced Text Processing (punctuation, NER, paragraphs)
from text_processor import TextProcessor, preload_text_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voxtral Streaming ASR",
    description="Voxtral Mini 3B streaming transcription service (Optimized)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Optimized
VOXTRAL_VLLM_URL = "http://localhost:4310"
CHUNK_DURATION_SECONDS = 1.5  # Reduced from 3s for lower latency
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = int(CHUNK_DURATION_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)

# VAD Configuration (fallback RMS-based)
VAD_ENERGY_THRESHOLD = 500  # RMS energy threshold for speech detection
VAD_SILENCE_DURATION = 0.8  # Seconds of silence before flush
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to process

# Neural VAD & Diarization Configuration
ENABLE_NEURAL_VAD = True  # Use Silero VAD instead of RMS energy
ENABLE_DIARIZATION = True  # Enable speaker diarization
HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token for pyannote

# Concurrency Configuration - Optimized for ~12 concurrent users
MAX_CONCURRENT_REQUESTS = 12
REQUEST_TIMEOUT = 30.0

# Request semaphore for concurrency control
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Connection pool for better performance
http_client: Optional[httpx.AsyncClient] = None

SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "nl": "Dutch",
    "it": "Italian",
}


@app.on_event("startup")
async def startup():
    """Initialize connection pool and preload models on startup."""
    global http_client
    http_client = httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=12),
    )
    logger.info("HTTP client pool initialized")

    # Preload text processing models (punctuation, NER, embeddings)
    logger.info("Preloading text processing models...")
    preload_text_models(
        enable_punctuation=True,
        enable_ner=True,
        enable_embeddings=True,
    )

    # Preload ASR enhancement models (VAD, diarization)
    logger.info("Preloading ASR enhancement models...")
    if ENABLE_DIARIZATION and HF_TOKEN:
        logger.info(f"Diarization enabled with HF token: {HF_TOKEN[:10]}...")
    elif ENABLE_DIARIZATION:
        logger.warning("Diarization enabled but HF_TOKEN not set - will fail on use")
    preload_models(
        enable_vad=ENABLE_NEURAL_VAD,
        enable_diarization=ENABLE_DIARIZATION,
        hf_token=HF_TOKEN,
    )


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global http_client
    if http_client:
        await http_client.aclose()
    logger.info("HTTP client pool closed")


def calculate_rms_energy(pcm_data: bytes) -> float:
    """Calculate RMS energy of PCM audio data for VAD."""
    if len(pcm_data) < 2:
        return 0.0
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def calculate_snr(pcm_data: bytes, noise_floor: float = 100.0) -> float:
    """Estimate Signal-to-Noise Ratio in dB."""
    if len(pcm_data) < 2:
        return 0.0
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    signal_power = np.mean(samples ** 2)
    if signal_power <= 0:
        return 0.0
    snr = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-10))
    return max(0.0, min(60.0, float(snr)))


def calculate_zero_crossing_rate(pcm_data: bytes) -> float:
    """Calculate zero crossing rate (indicator of speech vs noise)."""
    if len(pcm_data) < 4:
        return 0.0
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    signs = np.sign(samples)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / len(samples))


def detect_language_confidence(text: str) -> tuple[str, float]:
    """Simple language detection based on character patterns."""
    if not text:
        return "unknown", 0.0

    # Common words/patterns for supported languages
    patterns = {
        "en": (["the", "is", "are", "and", "of", "to", "in", "that", "it", "was"], 0.9),
        "de": (["der", "die", "das", "und", "ist", "ein", "eine", "zu", "den", "mit"], 0.85),
        "fr": (["le", "la", "les", "de", "et", "est", "un", "une", "que", "en"], 0.85),
        "es": (["el", "la", "los", "de", "que", "en", "un", "una", "es", "por"], 0.85),
        "pt": (["o", "a", "os", "de", "que", "em", "um", "uma", "para", "com"], 0.8),
        "it": (["il", "la", "di", "che", "è", "un", "una", "per", "non", "con"], 0.8),
        "nl": (["de", "het", "een", "van", "en", "in", "is", "dat", "op", "te"], 0.8),
    }

    text_lower = text.lower()
    words = text_lower.split()

    best_lang = "unknown"
    best_score = 0.0

    for lang, (common_words, base_conf) in patterns.items():
        matches = sum(1 for w in words if w in common_words)
        if len(words) > 0:
            score = (matches / len(words)) * base_conf
            if score > best_score:
                best_score = score
                best_lang = lang

    # Adjust confidence based on text length
    length_factor = min(1.0, len(words) / 10)
    confidence = best_score * length_factor

    return best_lang, round(confidence, 3)


# ============================================================================
# Text Enhancement: Punctuation, Sentences, Paragraphs, Timestamps
# ============================================================================

# Lazy-load punctuation model to avoid GPU memory at startup
_punctuation_model = None


def get_punctuation_model():
    """Lazy-load the punctuation restoration model."""
    global _punctuation_model
    if _punctuation_model is None:
        try:
            from deepmultilingualpunctuation import PunctuationModel
            _punctuation_model = PunctuationModel()
            logger.info("Punctuation model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load punctuation model: {e}")
            _punctuation_model = False  # Mark as failed, don't retry
    return _punctuation_model if _punctuation_model else None


class TextEnhancer:
    """
    Enhances raw ASR text with:
    - Punctuation restoration (AI-based)
    - Sentence boundary detection
    - Paragraph grouping
    - Word-level timestamp estimation
    - Speaker turn detection (based on pauses)
    """

    def __init__(self):
        self.sentences: list[dict] = []
        self.paragraphs: list[dict] = []
        self.current_paragraph_sentences: list[str] = []
        self.total_words = 0
        self.last_sentence_end_time = 0.0
        # Paragraph detection: new paragraph after long pause or topic shift
        self.paragraph_pause_threshold = 2.0  # seconds
        self.last_chunk_end_time = 0.0

    def enhance_text(
        self,
        raw_text: str,
        start_time: float,
        end_time: float,
        is_final: bool = False,
        apply_punctuation: bool = True,
    ) -> dict:
        """
        Enhance raw transcription text with punctuation and structure.

        Returns:
            {
                "raw": "hello how are you",
                "text": "Hello, how are you?",
                "sentences": [{"text": "Hello, how are you?", "start": 0.0, "end": 1.5}],
                "words": [{"word": "Hello", "start": 0.0, "end": 0.3}, ...],
                "is_question": True,
                "is_complete_sentence": True,
                "paragraph_break": False
            }
        """
        if not raw_text or not raw_text.strip():
            return {
                "raw": raw_text,
                "text": raw_text,
                "sentences": [],
                "words": [],
                "is_question": False,
                "is_complete_sentence": False,
                "paragraph_break": False,
            }

        # Apply punctuation restoration
        punctuated_text = raw_text
        if apply_punctuation:
            punctuated_text = self._restore_punctuation(raw_text)

        # Detect sentences
        sentences = self._detect_sentences(punctuated_text, start_time, end_time)

        # Estimate word timestamps
        words = self._estimate_word_timestamps(punctuated_text, start_time, end_time)

        # Check for paragraph break (long pause)
        paragraph_break = False
        if start_time - self.last_chunk_end_time > self.paragraph_pause_threshold:
            paragraph_break = True
            # Save current paragraph
            if self.current_paragraph_sentences:
                self.paragraphs.append({
                    "index": len(self.paragraphs),
                    "sentences": self.current_paragraph_sentences.copy(),
                    "text": " ".join(self.current_paragraph_sentences),
                })
                self.current_paragraph_sentences = []

        self.last_chunk_end_time = end_time

        # Add sentences to current paragraph
        for sent in sentences:
            self.current_paragraph_sentences.append(sent["text"])
            self.sentences.append(sent)

        # Detect question
        is_question = punctuated_text.rstrip().endswith("?")

        # Check if ends with sentence-ending punctuation
        is_complete = any(punctuated_text.rstrip().endswith(p) for p in ".!?")

        return {
            "raw": raw_text,
            "text": punctuated_text,
            "sentences": sentences,
            "words": words,
            "is_question": is_question,
            "is_complete_sentence": is_complete,
            "paragraph_break": paragraph_break,
        }

    def _restore_punctuation(self, text: str) -> str:
        """Restore punctuation using AI model."""
        model = get_punctuation_model()
        if model is None:
            # Fallback: basic capitalization
            return text.capitalize()

        try:
            result = model.restore_punctuation(text)
            return result
        except Exception as e:
            logger.warning(f"Punctuation restoration failed: {e}")
            return text.capitalize()

    def _detect_sentences(
        self, text: str, start_time: float, end_time: float
    ) -> list[dict]:
        """Split text into sentences with timestamp estimates."""
        import re

        # Split on sentence-ending punctuation
        sentence_pattern = r'(?<=[.!?])\s+'
        parts = re.split(sentence_pattern, text)
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            return []

        # Estimate timestamps proportionally by character count
        total_chars = sum(len(p) for p in parts)
        duration = end_time - start_time

        sentences = []
        current_time = start_time

        for i, part in enumerate(parts):
            if total_chars > 0:
                part_duration = (len(part) / total_chars) * duration
            else:
                part_duration = duration / len(parts)

            sentences.append({
                "index": len(self.sentences) + i,
                "text": part,
                "start": round(current_time, 3),
                "end": round(current_time + part_duration, 3),
                "word_count": len(part.split()),
            })
            current_time += part_duration

        return sentences

    def _estimate_word_timestamps(
        self, text: str, start_time: float, end_time: float
    ) -> list[dict]:
        """Estimate word-level timestamps based on text position."""
        words = text.split()
        if not words:
            return []

        duration = end_time - start_time
        # Average word duration (assuming ~150 WPM = 2.5 words/sec)
        avg_word_duration = duration / len(words)

        word_list = []
        current_time = start_time

        for word in words:
            # Adjust duration based on word length
            length_factor = len(word) / 5.0  # normalize to ~5 char average
            word_duration = avg_word_duration * max(0.5, min(2.0, length_factor))

            word_list.append({
                "word": word,
                "start": round(current_time, 3),
                "end": round(current_time + word_duration, 3),
                "confidence": 0.85,  # Estimated confidence
            })
            current_time += word_duration

        # Adjust last word to match end_time
        if word_list:
            word_list[-1]["end"] = round(end_time, 3)

        return word_list

    def get_formatted_transcript(self) -> str:
        """Get full transcript with paragraph breaks."""
        paragraphs_text = []

        # Add completed paragraphs
        for para in self.paragraphs:
            paragraphs_text.append(para["text"])

        # Add current paragraph in progress
        if self.current_paragraph_sentences:
            paragraphs_text.append(" ".join(self.current_paragraph_sentences))

        return "\n\n".join(paragraphs_text)

    def get_all_sentences(self) -> list[dict]:
        """Get all sentences with timestamps."""
        return self.sentences

    def finalize(self) -> dict:
        """Finalize and return complete structure."""
        # Save any remaining paragraph
        if self.current_paragraph_sentences:
            self.paragraphs.append({
                "index": len(self.paragraphs),
                "sentences": self.current_paragraph_sentences.copy(),
                "text": " ".join(self.current_paragraph_sentences),
            })

        return {
            "paragraphs": self.paragraphs,
            "sentences": self.sentences,
            "total_sentences": len(self.sentences),
            "total_paragraphs": len(self.paragraphs),
        }


class VoxtralStreamingProcessor:
    """Handles chunked audio processing with VAD for near-real-time transcription."""

    def __init__(
        self,
        language: str = "auto",
        enable_punctuation: bool = True,
        enable_neural_vad: bool = True,
        enable_diarization: bool = True,
    ):
        self.language = language
        self.enable_punctuation = enable_punctuation
        self.audio_buffer = bytearray()
        self.full_transcript: list[str] = []  # Raw transcript chunks
        self.last_speech_time = time.time()
        self.speech_detected = False
        self.pending_requests = 0
        self.request_queue: deque = deque(maxlen=10)
        # Statistics
        self.start_time = time.time()
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self.chunk_count = 0
        self.detected_language = "auto"
        self.word_count = 0
        # Advanced Text Processing (punctuation, NER, paragraphs)
        self.text_processor = TextProcessor(
            enable_punctuation=enable_punctuation,
            enable_ner=True,
            enable_paragraphs=True,
        )
        self.enhanced_transcript: list[str] = []  # Punctuated transcript
        # Legacy text enhancer (keep for compatibility, but use new processor)
        self.text_enhancer = TextEnhancer()
        # Timestamp tracking
        self.session_start_time = time.time()
        self.current_audio_position = 0.0  # Track position in audio stream
        # ASR Enhancers (Neural VAD + Diarization)
        self.asr_enhancer = ASREnhancer(
            enable_vad=enable_neural_vad,
            enable_diarization=enable_diarization,
            hf_token=HF_TOKEN,
            sample_rate=SAMPLE_RATE,
        )
        self.enable_neural_vad = enable_neural_vad
        self.enable_diarization = enable_diarization
        # Diarization results
        self.speaker_segments: list[dict] = []
        self.current_speaker: Optional[str] = None

    async def process_chunk(self, audio_data: bytes) -> Optional[dict]:
        """Process an audio chunk with VAD and return transcription."""
        self.audio_buffer.extend(audio_data)

        # Calculate energy for VAD (fallback)
        energy = calculate_rms_energy(audio_data)
        current_time = time.time()

        # Update speech detection state
        if energy > VAD_ENERGY_THRESHOLD:
            self.speech_detected = True
            self.last_speech_time = current_time

        # Check if we should process
        silence_duration = current_time - self.last_speech_time
        buffer_duration = len(self.audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        should_process = False
        is_final_chunk = False

        # Process conditions:
        # 1. Buffer full and speech detected
        # 2. Silence detected after speech (flush)
        if len(self.audio_buffer) >= CHUNK_SIZE_BYTES and self.speech_detected:
            should_process = True
        elif (silence_duration > VAD_SILENCE_DURATION and
              self.speech_detected and
              buffer_duration > MIN_SPEECH_DURATION):
            should_process = True
            is_final_chunk = True
            self.speech_detected = False

        if not should_process:
            return None

        # Extract chunk for processing
        if is_final_chunk:
            chunk = bytes(self.audio_buffer)
            self.audio_buffer.clear()
        else:
            chunk = bytes(self.audio_buffer[:CHUNK_SIZE_BYTES])
            self.audio_buffer = self.audio_buffer[CHUNK_SIZE_BYTES:]

        # Calculate audio metrics
        chunk_energy = calculate_rms_energy(chunk)

        # Neural VAD check (if enabled)
        neural_vad_result = None
        if self.enable_neural_vad:
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            neural_vad_result = self.asr_enhancer.process(
                audio_array,
                start_offset=self.current_audio_position,
            )
            # Skip if neural VAD says no speech
            if not neural_vad_result.is_speech:
                return None
        elif chunk_energy < VAD_ENERGY_THRESHOLD * 0.5:
            # Fallback: Skip if mostly silence (energy-based)
            return None

        # Calculate additional metrics
        snr = calculate_snr(chunk)
        zcr = calculate_zero_crossing_rate(chunk)
        chunk_duration = len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        self.total_audio_duration += chunk_duration
        self.chunk_count += 1

        # Send to Voxtral vLLM with concurrency control
        proc_start = time.time()
        result = await self._transcribe(chunk)
        proc_time = time.time() - proc_start
        self.total_processing_time += proc_time

        if result:
            raw_text = result.get("text", "")
            usage = result.get("usage", {})

            if raw_text:
                self.full_transcript.append(raw_text)
                words = raw_text.split()
                self.word_count += len(words)

                # Calculate timestamps for this chunk
                chunk_start_time = self.current_audio_position
                chunk_end_time = chunk_start_time + chunk_duration
                self.current_audio_position = chunk_end_time

                # Advanced text processing (punctuation, NER, paragraphs)
                processed = self.text_processor.process(
                    raw_text,
                    start_time=chunk_start_time,
                    end_time=chunk_end_time,
                    speaker=neural_vad_result.current_speaker if neural_vad_result else None,
                    is_final=is_final_chunk,
                )
                self.enhanced_transcript.append(processed.text)

                # Keep legacy enhanced dict for compatibility
                enhanced = {
                    "text": processed.text,
                    "sentences": processed.sentences,
                    "words": self.text_enhancer._estimate_word_timestamps(processed.text, chunk_start_time, chunk_end_time),
                    "is_question": processed.is_question,
                    "is_complete_sentence": processed.is_complete_sentence,
                    "paragraph_break": processed.paragraph_break,
                }

                # Detect language if auto
                if self.language == "auto":
                    lang, lang_conf = detect_language_confidence(self.get_full_transcript())
                    self.detected_language = lang
                else:
                    lang = self.language
                    lang_conf = 1.0

                # Calculate real-time factor (RTF)
                rtf = proc_time / chunk_duration if chunk_duration > 0 else 0

                # Words per minute estimate
                session_duration = time.time() - self.start_time
                wpm = (self.word_count / session_duration * 60) if session_duration > 0 else 0

                # Build diarization data if available
                diarization_data = None
                if neural_vad_result and neural_vad_result.speaker_segments:
                    # Update speaker tracking
                    for seg in neural_vad_result.speaker_segments:
                        self.speaker_segments.append({
                            "speaker": seg.speaker,
                            "start": seg.start,
                            "end": seg.end,
                        })
                    self.current_speaker = neural_vad_result.current_speaker
                    diarization_data = {
                        "current_speaker": self.current_speaker,
                        "segments": [
                            {"speaker": s.speaker, "start": s.start, "end": s.end}
                            for s in neural_vad_result.speaker_segments
                        ],
                        "speakers": neural_vad_result.speakers,
                        "processing_ms": round(neural_vad_result.diarization_time_ms, 1),
                    }

                # Build VAD data if available
                vad_data = None
                if neural_vad_result:
                    vad_data = {
                        "is_speech": neural_vad_result.is_speech,
                        "speech_ratio": round(neural_vad_result.speech_ratio, 2),
                        "segments": [
                            {"start": s.start, "end": s.end}
                            for s in neural_vad_result.vad_segments
                        ],
                        "processing_ms": round(neural_vad_result.vad_time_ms, 1),
                    }

                return {
                    "type": "final" if is_final_chunk else "partial",
                    # Raw and enhanced text
                    "text": enhanced["text"],  # Punctuated text
                    "raw_text": raw_text,  # Original unpunctuated
                    "full_transcript": self.get_enhanced_transcript(),
                    "raw_transcript": self.get_full_transcript(),
                    # Timestamps
                    "timing": {
                        "start": round(chunk_start_time, 3),
                        "end": round(chunk_end_time, 3),
                        "duration": round(chunk_duration, 3),
                    },
                    # Sentences with timestamps
                    "sentences": enhanced["sentences"],
                    # Word-level timestamps
                    "words": enhanced["words"],
                    # Text structure
                    "structure": {
                        "is_question": enhanced["is_question"],
                        "is_complete_sentence": enhanced["is_complete_sentence"],
                        "paragraph_break": enhanced["paragraph_break"],
                        "sentence_count": len(enhanced["sentences"]),
                    },
                    # Named entities (persons, dates, numbers, etc.)
                    "entities": processed.entities,
                    # Paragraphs (semantic grouping)
                    "paragraphs": self.text_processor.get_paragraphs(),
                    # Speaker diarization (who is speaking)
                    "diarization": diarization_data,
                    # Neural VAD results
                    "vad": vad_data,
                    # Audio metrics
                    "audio": {
                        "energy_rms": round(chunk_energy, 1),
                        "snr_db": round(snr, 1),
                        "zero_crossing_rate": round(zcr, 4),
                        "duration_sec": round(chunk_duration, 2),
                        "sample_rate": SAMPLE_RATE,
                    },
                    # Processing metrics
                    "processing": {
                        "latency_ms": round(proc_time * 1000, 1),
                        "rtf": round(rtf, 3),  # Real-time factor (<1 = faster than real-time)
                        "chunk_index": self.chunk_count,
                    },
                    # Language detection
                    "language": {
                        "requested": self.language,
                        "detected": lang,
                        "confidence": lang_conf,
                    },
                    # Transcript stats
                    "stats": {
                        "word_count": self.word_count,
                        "char_count": len(self.get_full_transcript()),
                        "wpm": round(wpm, 1),
                        "total_audio_sec": round(self.total_audio_duration, 2),
                        "total_proc_ms": round(self.total_processing_time * 1000, 1),
                        "total_sentences": len(self.text_processor.sentences),
                        "total_paragraphs": len(self.text_processor.get_paragraphs()),
                        "total_speakers": len(neural_vad_result.speakers) if neural_vad_result else 0,
                        "total_entities": len(self.text_processor.all_entities),
                        "entities_summary": self.text_processor.get_entities_summary(),
                    },
                    # NER processing time
                    "ner": {
                        "processing_ms": round(processed.ner_ms, 1),
                        "entity_count": len(processed.entities),
                    },
                    # Usage from vLLM (if available)
                    "usage": usage,
                }

        return None

    async def _transcribe(self, chunk: bytes) -> Optional[dict]:
        """Send audio to vLLM for transcription with concurrency control."""
        async with request_semaphore:
            try:
                wav_data = self._create_wav(chunk)
                files = {"file": ("chunk.wav", wav_data, "audio/wav")}
                data = {"model": "mistralai/Voxtral-Mini-3B-2507"}

                if self.language != "auto":
                    data["language"] = self.language

                response = await http_client.post(
                    f"{VOXTRAL_VLLM_URL}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                )

                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.error(f"Transcription error: {e}")

        return None

    async def flush(self) -> Optional[dict]:
        """Process remaining audio in buffer."""
        if len(self.audio_buffer) < SAMPLE_RATE * BYTES_PER_SAMPLE * MIN_SPEECH_DURATION:
            return None

        chunk = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        # Skip if mostly silence
        chunk_energy = calculate_rms_energy(chunk)
        if chunk_energy < VAD_ENERGY_THRESHOLD * 0.5:
            return None

        chunk_duration = len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        self.total_audio_duration += chunk_duration
        self.chunk_count += 1

        proc_start = time.time()
        result = await self._transcribe(chunk)
        proc_time = time.time() - proc_start
        self.total_processing_time += proc_time

        if result:
            raw_text = result.get("text", "")
            usage = result.get("usage", {})

            if raw_text:
                self.full_transcript.append(raw_text)
                words = raw_text.split()
                self.word_count += len(words)

                # Calculate timestamps for flush chunk
                chunk_start_time = self.current_audio_position
                chunk_end_time = chunk_start_time + chunk_duration
                self.current_audio_position = chunk_end_time

                # Enhance text (final flush)
                enhanced = self.text_enhancer.enhance_text(
                    raw_text,
                    start_time=chunk_start_time,
                    end_time=chunk_end_time,
                    is_final=True,
                    apply_punctuation=self.enable_punctuation,
                )
                self.enhanced_transcript.append(enhanced["text"])

                lang, lang_conf = detect_language_confidence(self.get_full_transcript())
                rtf = proc_time / chunk_duration if chunk_duration > 0 else 0
                session_duration = time.time() - self.start_time
                wpm = (self.word_count / session_duration * 60) if session_duration > 0 else 0

                # Get final structure
                final_structure = self.text_enhancer.finalize()

                return {
                    "type": "final",
                    "text": enhanced["text"],
                    "raw_text": raw_text,
                    "full_transcript": self.get_enhanced_transcript(),
                    "raw_transcript": self.get_full_transcript(),
                    "formatted_transcript": self.get_formatted_transcript(),
                    "timing": {
                        "start": round(chunk_start_time, 3),
                        "end": round(chunk_end_time, 3),
                        "duration": round(chunk_duration, 3),
                    },
                    "sentences": enhanced["sentences"],
                    "words": enhanced["words"],
                    "structure": {
                        "is_question": enhanced["is_question"],
                        "is_complete_sentence": enhanced["is_complete_sentence"],
                        "paragraph_break": enhanced["paragraph_break"],
                        "sentence_count": len(enhanced["sentences"]),
                        "all_sentences": final_structure["sentences"],
                        "all_paragraphs": final_structure["paragraphs"],
                    },
                    "audio": {
                        "energy_rms": round(chunk_energy, 1),
                        "snr_db": round(calculate_snr(chunk), 1),
                        "zero_crossing_rate": round(calculate_zero_crossing_rate(chunk), 4),
                        "duration_sec": round(chunk_duration, 2),
                        "sample_rate": SAMPLE_RATE,
                    },
                    "processing": {
                        "latency_ms": round(proc_time * 1000, 1),
                        "rtf": round(rtf, 3),
                        "chunk_index": self.chunk_count,
                    },
                    "language": {
                        "requested": self.language,
                        "detected": lang,
                        "confidence": lang_conf,
                    },
                    "stats": {
                        "word_count": self.word_count,
                        "char_count": len(self.get_full_transcript()),
                        "wpm": round(wpm, 1),
                        "total_audio_sec": round(self.total_audio_duration, 2),
                        "total_proc_ms": round(self.total_processing_time * 1000, 1),
                        "total_sentences": final_structure["total_sentences"],
                        "total_paragraphs": final_structure["total_paragraphs"],
                    },
                    "usage": usage,
                }

        return None

    def _create_wav(self, pcm_data: bytes) -> bytes:
        """Create WAV file from PCM data."""
        channels = 1
        sample_width = 2
        frame_rate = SAMPLE_RATE

        data_size = len(pcm_data)
        file_size = data_size + 36

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            file_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            channels,
            frame_rate,
            frame_rate * channels * sample_width,
            channels * sample_width,
            sample_width * 8,
            b"data",
            data_size,
        )

        return header + pcm_data

    def get_full_transcript(self) -> str:
        """Get the complete raw transcript (without punctuation)."""
        return " ".join(self.full_transcript)

    def get_enhanced_transcript(self) -> str:
        """Get the complete transcript with punctuation and formatting."""
        return " ".join(self.enhanced_transcript)

    def get_formatted_transcript(self) -> str:
        """Get transcript with paragraph breaks."""
        return self.text_enhancer.get_formatted_transcript()

    def get_transcript_structure(self) -> dict:
        """Get complete transcript with sentences and paragraphs."""
        return self.text_enhancer.finalize()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve optimized demo page with debug window."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Voxtral Streaming ASR - Debug</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
               margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .panel { background: #16213e; border-radius: 12px; padding: 20px; }
        h1 { color: #00d4ff; margin: 0 0 5px 0; font-size: 24px; }
        h2 { color: #00d4ff; margin: 0 0 15px 0; font-size: 16px; border-bottom: 1px solid #0f3460; padding-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 15px; font-size: 13px; }
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
        #transcript { min-height: 150px; border: 2px solid #0f3460; border-radius: 8px;
                     padding: 15px; margin: 15px 0; background: #0f3460; font-size: 15px; line-height: 1.5; }
        .partial { color: #ffd166; font-style: italic; }
        .speaker-label { color: #40916c; font-weight: bold; margin-right: 6px; }
        .speaker-1 { color: #40916c; }
        .speaker-2 { color: #f77f00; }
        .speaker-3 { color: #9d4edd; }
        .speaker-4 { color: #00d4ff; }
        .sentence-block { margin: 4px 0; }
        /* Entity highlighting */
        .entity { padding: 1px 4px; border-radius: 3px; margin: 0 1px; font-weight: 500; }
        .entity-person { background: #2d6a4f; color: #95d5b2; }
        .entity-date { background: #4a3f15; color: #ffd166; }
        .entity-time { background: #3d3d15; color: #ffe066; }
        .entity-number { background: #1b4332; color: #b7e4c7; }
        .entity-money { background: #5c4d1a; color: #ffd700; }
        .entity-organization { background: #0f3460; color: #00d4ff; }
        .entity-location { background: #4a1515; color: #f8b4b4; }
        .entity-phone { background: #2d3a4f; color: #a8dadc; }
        .entity-percentage { background: #3d2d4f; color: #c8b6ff; }
        /* Entity summary panel */
        .entities-panel { background: #0f3460; border-radius: 8px; padding: 12px; margin: 10px 0; }
        .entities-panel h3 { color: #00d4ff; margin: 0 0 10px 0; font-size: 13px; }
        .entity-group { margin: 6px 0; }
        .entity-type { color: #888; font-size: 11px; text-transform: uppercase; margin-right: 8px; }
        .entity-list { display: inline; }
        .entity-item { display: inline-block; margin: 2px 4px 2px 0; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }
        .stat-box { background: #0f3460; padding: 10px; border-radius: 6px; text-align: center; }
        .stat-label { font-size: 11px; color: #888; margin-bottom: 4px; }
        .stat-value { font-size: 18px; color: #00d4ff; font-weight: bold; }
        .energy-bar { height: 8px; background: #0f3460; border-radius: 4px; overflow: hidden; margin-top: 5px; }
        .energy-level { height: 100%; background: linear-gradient(90deg, #2d6a4f, #40916c, #ffd166, #f77f00); transition: width 0.1s; }

        /* Debug Panel Styles */
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
        .log-data { color: #ffd166; margin-left: 20px; display: block; white-space: pre-wrap; word-break: break-all; }

        /* Protocol Info */
        .protocol-info { background: #0f3460; border-radius: 8px; padding: 15px; margin-top: 15px; font-size: 12px; }
        .protocol-info h3 { color: #00d4ff; margin: 0 0 10px 0; font-size: 14px; }
        .protocol-info code { background: #1a1a2e; padding: 2px 6px; border-radius: 3px; color: #ffd166; }
        .protocol-info pre { background: #1a1a2e; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 10px 0; }

        @media (max-width: 1000px) {
            .layout { grid-template-columns: 1fr; }
            .debug-panel { height: 500px; }
        }
    </style>
</head>
<body>
    <div class="layout">
        <!-- Left Panel: Controls & Transcript -->
        <div class="panel">
            <h1>Voxtral Mini 3B - Streaming ASR</h1>
            <p class="subtitle">Real-time speech-to-text with VAD | Apache 2.0 License</p>

            <div id="status" class="status disconnected">Disconnected</div>

            <div class="controls">
                <select id="language">
                    <option value="auto">Auto-detect</option>
                    <option value="en">English</option>
                    <option value="de">German</option>
                    <option value="fr">French</option>
                    <option value="es">Spanish</option>
                    <option value="pt">Portuguese</option>
                    <option value="hi">Hindi</option>
                    <option value="nl">Dutch</option>
                    <option value="it">Italian</option>
                </select>
                <button id="connectBtn" class="btn-primary" onclick="connect()">Connect</button>
                <button id="startBtn" class="btn-success" onclick="startRecording()" disabled>Start Recording</button>
                <button id="stopBtn" class="btn-danger" onclick="stopRecording()" disabled>Stop Recording</button>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">ENERGY (RMS)</div>
                    <div class="stat-value" id="energyVal">0</div>
                    <div class="energy-bar"><div id="energyBar" class="energy-level" style="width: 0%"></div></div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">SNR (dB)</div>
                    <div class="stat-value" id="snrVal">-</div>
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
                    <div class="stat-label">LANGUAGE</div>
                    <div class="stat-value" id="langVal">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">CONFIDENCE</div>
                    <div class="stat-value" id="langConf">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">WPM</div>
                    <div class="stat-value" id="wpmVal">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">WORDS</div>
                    <div class="stat-value" id="wordCount">0</div>
                </div>
            </div>
            <div class="stats" style="grid-template-columns: repeat(3, 1fr);">
                <div class="stat-box">
                    <div class="stat-label">CHUNKS SENT</div>
                    <div class="stat-value" id="chunkCount">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">BYTES SENT</div>
                    <div class="stat-value" id="bytesSent">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">AUDIO TIME</div>
                    <div class="stat-value" id="audioTime">0s</div>
                </div>
            </div>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">SENTENCES</div>
                    <div class="stat-value" id="sentenceCount">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">PARAGRAPHS</div>
                    <div class="stat-value" id="paragraphCount">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">TIME POS</div>
                    <div class="stat-value" id="timePos">0.0s</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">TYPE</div>
                    <div class="stat-value" id="sentenceType">-</div>
                </div>
            </div>
            <div class="stats">
                <div class="stat-box" style="background: #1b4332;">
                    <div class="stat-label">SPEAKER</div>
                    <div class="stat-value" id="currentSpeaker">-</div>
                </div>
                <div class="stat-box" style="background: #1b4332;">
                    <div class="stat-label">SPEAKERS</div>
                    <div class="stat-value" id="speakerCount">0</div>
                </div>
                <div class="stat-box" style="background: #0f3460;">
                    <div class="stat-label">VAD</div>
                    <div class="stat-value" id="vadStatus">-</div>
                </div>
                <div class="stat-box" style="background: #0f3460;">
                    <div class="stat-label">SPEECH %</div>
                    <div class="stat-value" id="speechRatio">-</div>
                </div>
            </div>

            <h2>Transcript</h2>
            <div id="transcript"></div>

            <!-- Entities Panel -->
            <div class="entities-panel" id="entitiesPanel" style="display: none;">
                <h3>Detected Entities</h3>
                <div id="entitiesSummary"></div>
            </div>

            <!-- Protocol Info -->
            <div class="protocol-info">
                <h3>WebSocket Protocol</h3>
                <p><strong>Endpoint:</strong> <code>wss://voxtral.dudoxx.com/asr?language=auto</code></p>
                <p><strong>Audio Format:</strong> PCM Int16, 16kHz, Mono</p>
                <p><strong>Send:</strong> Binary ArrayBuffer (Int16Array.buffer)</p>
                <p><strong>Receive:</strong> JSON with punctuation, timestamps, sentences</p>
                <pre style="font-size:10px; max-height:200px; overflow-y:auto;">{
  "type": "partial|final",
  "text": "Hello, how are you?",
  "raw_text": "hello how are you",
  "full_transcript": "Hello, how are you?",
  "timing": {
    "start": 0.0, "end": 1.5, "duration": 1.5
  },
  "sentences": [{
    "index": 0, "text": "Hello, how are you?",
    "start": 0.0, "end": 1.5, "word_count": 4
  }],
  "words": [{
    "word": "Hello,", "start": 0.0, "end": 0.3
  }, ...],
  "structure": {
    "is_question": true,
    "is_complete_sentence": true,
    "paragraph_break": false
  },
  "audio": { "energy_rms": 1234.5, "snr_db": 25.3 },
  "processing": { "latency_ms": 180.5, "rtf": 0.12 },
  "language": { "detected": "en", "confidence": 0.85 },
  "stats": { "word_count": 4, "wpm": 145.2, "total_sentences": 1 }
}</pre>
            </div>
        </div>

        <!-- Right Panel: Debug Log -->
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
        let lastSendTime = 0;
        let messageId = 0;

        // Speaker tracking for transcript
        let sentenceHistory = [];  // [{text, speaker, start, end, entities}]
        let lastSpeaker = null;
        let allEntities = [];  // Track all detected entities

        // Entity highlighting helper
        const entityColors = {
            'PERSON': 'entity-person',
            'DATE': 'entity-date',
            'TIME': 'entity-time',
            'NUMBER': 'entity-number',
            'MONEY': 'entity-money',
            'ORGANIZATION': 'entity-organization',
            'LOCATION': 'entity-location',
            'PHONE NUMBER': 'entity-phone',
            'PERCENTAGE': 'entity-percentage',
        };

        function highlightEntities(text, entities) {
            if (!entities || entities.length === 0) return text;

            // Sort entities by position (descending) to replace from end to start
            const sorted = [...entities].sort((a, b) => b.start - a.start);

            let result = text;
            for (const entity of sorted) {
                const colorClass = entityColors[entity.label] || 'entity-person';
                const highlighted = '<span class="entity ' + colorClass + '" title="' + entity.label + '">' + entity.text + '</span>';
                // Try to find and replace the entity text
                const idx = result.toLowerCase().indexOf(entity.text.toLowerCase());
                if (idx !== -1) {
                    result = result.substring(0, idx) + highlighted + result.substring(idx + entity.text.length);
                }
            }
            return result;
        }

        // Debug logging
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

            entry.innerHTML = '<span class="log-time">' + time + '</span>' +
                             '<span class="' + dirClass + '">' + dirLabel + '</span> ' +
                             type + dataStr;

            debugLog.appendChild(entry);

            if (document.getElementById('autoScroll').checked) {
                debugLog.scrollTop = debugLog.scrollHeight;
            }
        }

        function clearLog() {
            document.getElementById('debugLog').innerHTML = '';
            log('Log cleared', 'info');
        }

        function copyLog() {
            const text = document.getElementById('debugLog').innerText;
            navigator.clipboard.writeText(text).then(() => {
                log('Log copied to clipboard', 'info');
            });
        }

        function updateStatus(status) {
            const el = document.getElementById('status');
            el.className = 'status ' + status;
            const texts = { connected: 'Connected', disconnected: 'Disconnected', recording: 'Recording...' };
            el.textContent = texts[status] || status;
        }

        function updateEnergy(energy) {
            const normalized = Math.min(100, (energy / 5000) * 100);
            document.getElementById('energyVal').textContent = Math.round(energy);
            document.getElementById('energyBar').style.width = normalized + '%';
        }

        function formatBytes(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
            return (bytes/1024/1024).toFixed(2) + ' MB';
        }

        function connect() {
            const lang = document.getElementById('language').value;
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + location.host + '/asr?language=' + lang;

            log('Connecting to WebSocket', 'info', wsUrl);

            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                log('WebSocket OPEN', 'recv', { readyState: ws.readyState, protocol: ws.protocol });
                updateStatus('connected');
                document.getElementById('connectBtn').disabled = true;
                document.getElementById('startBtn').disabled = false;
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                log('Message received', 'recv', data);

                const transcript = document.getElementById('transcript');

                if (data.type === 'config') {
                    log('Server config received', 'info', {
                        language: data.language,
                        chunk_duration: data.chunk_duration,
                        model: data.model,
                        vad_enabled: data.vad_enabled
                    });
                }
                else if (data.type === 'partial' || data.type === 'final') {
                    // Get current speaker from diarization
                    const currentSpeaker = data.diarization?.current_speaker || null;
                    const currentText = data.text || '';

                    // Get entities for this chunk
                    const chunkEntities = data.entities || [];

                    // Add sentence to history with speaker and entities info
                    if (data.sentences && data.sentences.length > 0) {
                        const sentence = data.sentences[data.sentences.length - 1];
                        // Update or add the current sentence
                        if (sentenceHistory.length === 0 ||
                            sentenceHistory[sentenceHistory.length - 1].index !== sentence.index) {
                            sentenceHistory.push({
                                index: sentence.index,
                                text: sentence.text,
                                speaker: currentSpeaker,
                                start: sentence.start,
                                end: sentence.end,
                                entities: chunkEntities
                            });
                        } else {
                            // Update existing sentence
                            sentenceHistory[sentenceHistory.length - 1].text = sentence.text;
                            sentenceHistory[sentenceHistory.length - 1].speaker = currentSpeaker;
                            sentenceHistory[sentenceHistory.length - 1].entities = chunkEntities;
                        }
                    }

                    // Build transcript with speaker labels and entity highlighting
                    let transcriptHtml = '';
                    let prevSpeaker = null;
                    for (const sent of sentenceHistory) {
                        const speaker = sent.speaker;
                        const speakerNum = speaker ? speaker.replace('Speaker ', '') : '1';
                        const speakerClass = 'speaker-' + Math.min(parseInt(speakerNum) || 1, 4);

                        // Show speaker label when speaker changes
                        if (speaker && speaker !== prevSpeaker) {
                            transcriptHtml += '<span class="speaker-label ' + speakerClass + '">[' + speaker + ']</span>';
                        }

                        // Highlight entities in sentence text
                        const highlightedText = highlightEntities(sent.text, sent.entities);
                        transcriptHtml += '<span class="sentence-text">' + highlightedText + '</span> ';
                        prevSpeaker = speaker;
                    }

                    // Add partial text indicator
                    if (data.type === 'partial' && currentText) {
                        transcriptHtml += '<span class="partial">' + currentText + '</span>';
                    }

                    transcript.innerHTML = transcriptHtml;

                    // Update audio metrics
                    if (data.audio) {
                        updateEnergy(data.audio.energy_rms || 0);
                        document.getElementById('snrVal').textContent = data.audio.snr_db || '-';
                    }

                    // Update processing metrics
                    if (data.processing) {
                        document.getElementById('latency').textContent = data.processing.latency_ms || '-';
                        document.getElementById('rtfVal').textContent = data.processing.rtf || '-';
                    }

                    // Update language info
                    if (data.language) {
                        document.getElementById('langVal').textContent = (data.language.detected || '-').toUpperCase();
                        const conf = data.language.confidence;
                        document.getElementById('langConf').textContent = conf ? (conf * 100).toFixed(0) + '%' : '-';
                    }

                    // Update stats
                    if (data.stats) {
                        document.getElementById('wpmVal').textContent = data.stats.wpm || '-';
                        document.getElementById('wordCount').textContent = data.stats.word_count || 0;
                        const audioSec = data.stats.total_audio_sec || 0;
                        document.getElementById('audioTime').textContent = audioSec.toFixed(1) + 's';
                        document.getElementById('sentenceCount').textContent = data.stats.total_sentences || 0;
                        document.getElementById('paragraphCount').textContent = data.stats.total_paragraphs || 0;
                    }

                    // Update timing info
                    if (data.timing) {
                        document.getElementById('timePos').textContent = data.timing.end.toFixed(1) + 's';
                    }

                    // Update sentence type indicator
                    if (data.structure) {
                        let typeText = '-';
                        if (data.structure.is_question) typeText = '?';
                        else if (data.structure.is_complete_sentence) typeText = '.';
                        else typeText = '...';
                        if (data.structure.paragraph_break) typeText = 'P ' + typeText;
                        document.getElementById('sentenceType').textContent = typeText;
                    }

                    // Update diarization (speaker) info
                    if (data.diarization) {
                        document.getElementById('currentSpeaker').textContent = data.diarization.current_speaker || '-';
                        document.getElementById('speakerCount').textContent = data.diarization.speakers ? data.diarization.speakers.length : 0;
                    }

                    // Update VAD info
                    if (data.vad) {
                        document.getElementById('vadStatus').textContent = data.vad.is_speech ? 'SPEECH' : 'SILENT';
                        document.getElementById('speechRatio').textContent = Math.round(data.vad.speech_ratio * 100) + '%';
                    }

                    // Update total speakers from stats
                    if (data.stats && data.stats.total_speakers !== undefined) {
                        document.getElementById('speakerCount').textContent = data.stats.total_speakers;
                    }

                    // Update entities panel
                    if (data.stats && data.stats.entities_summary) {
                        const summary = data.stats.entities_summary;
                        const panel = document.getElementById('entitiesPanel');
                        const container = document.getElementById('entitiesSummary');

                        if (Object.keys(summary).length > 0) {
                            panel.style.display = 'block';
                            let html = '';
                            const entityColors = {
                                'PERSON': 'entity-person',
                                'DATE': 'entity-date',
                                'TIME': 'entity-time',
                                'NUMBER': 'entity-number',
                                'MONEY': 'entity-money',
                                'ORGANIZATION': 'entity-organization',
                                'LOCATION': 'entity-location',
                                'PHONE NUMBER': 'entity-phone',
                                'PERCENTAGE': 'entity-percentage',
                            };

                            for (const [type, items] of Object.entries(summary)) {
                                const colorClass = entityColors[type] || 'entity-person';
                                html += '<div class="entity-group">';
                                html += '<span class="entity-type">' + type + ':</span>';
                                html += '<span class="entity-list">';
                                items.forEach(item => {
                                    html += '<span class="entity-item entity ' + colorClass + '">' + item + '</span>';
                                });
                                html += '</span></div>';
                            }
                            container.innerHTML = html;
                        }
                    }
                }
                else if (data.type === 'error') {
                    log('Server error', 'error', data.message);
                }
            };

            ws.onclose = (event) => {
                log('WebSocket CLOSE', 'recv', { code: event.code, reason: event.reason, wasClean: event.wasClean });
                updateStatus('disconnected');
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('startBtn').disabled = true;
                stopRecording();
            };

            ws.onerror = (err) => {
                log('WebSocket ERROR', 'error', err.message || 'Connection error');
            };
        }

        async function startRecording() {
            if (isRecording) return;

            log('Requesting microphone access', 'info', {
                sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true
            });

            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                });

                const tracks = mediaStream.getAudioTracks();
                log('Microphone access granted', 'info', {
                    track: tracks[0].label,
                    settings: tracks[0].getSettings()
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                log('AudioContext created', 'info', { sampleRate: audioContext.sampleRate, state: audioContext.state });

                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(2048, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcm = new Int16Array(inputData.length);
                    let energy = 0;

                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        energy += pcm[i] * pcm[i];
                    }

                    energy = Math.sqrt(energy / inputData.length);
                    updateEnergy(energy);

                    const buffer = pcm.buffer;
                    lastSendTime = Date.now();
                    ws.send(buffer);

                    chunkCount++;
                    bytesSent += buffer.byteLength;
                    document.getElementById('chunkCount').textContent = chunkCount;
                    document.getElementById('bytesSent').textContent = formatBytes(bytesSent);

                    // Log every 10th chunk to avoid spam
                    if (chunkCount % 10 === 0) {
                        log('Audio chunk #' + chunkCount, 'send', {
                            bytes: buffer.byteLength,
                            samples: pcm.length,
                            duration_ms: Math.round(pcm.length / 16000 * 1000),
                            energy: Math.round(energy),
                            total_sent: formatBytes(bytesSent)
                        });
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                chunkCount = 0;
                bytesSent = 0;
                sentenceHistory = [];  // Reset speaker tracking
                lastSpeaker = null;
                allEntities = [];  // Reset entities
                document.getElementById('entitiesPanel').style.display = 'none';
                document.getElementById('entitiesSummary').innerHTML = '';
                updateStatus('recording');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;

                log('Recording started', 'info', { bufferSize: 2048, channels: 1, format: 'PCM Int16' });

            } catch (err) {
                log('Microphone error', 'error', err.message);
                alert('Error accessing microphone: ' + err.message);
            }
        }

        function stopRecording() {
            if (!isRecording && !processor) return;

            isRecording = false;
            log('Stopping recording', 'info', { totalChunks: chunkCount, totalBytes: formatBytes(bytesSent) });

            if (ws && ws.readyState === WebSocket.OPEN) {
                log('Sending empty buffer (end signal)', 'send', { bytes: 0 });
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

        // Initial log
        log('Page loaded', 'info', {
            userAgent: navigator.userAgent.substr(0, 50) + '...',
            location: location.origin
        });
    </script>
</body>
</html>
    """


@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket, language: str = "auto"):
    """WebSocket endpoint for streaming ASR with VAD."""
    await websocket.accept()
    logger.info(f"WebSocket connected, language: {language}")

    processor = VoxtralStreamingProcessor(language=language)

    try:
        await websocket.send_json({
            "type": "config",
            "language": language,
            "chunk_duration": CHUNK_DURATION_SECONDS,
            "model": "mistralai/Voxtral-Mini-3B-2507",
            "vad_enabled": True,
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
    finally:
        logger.info("WebSocket cleanup complete")


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="auto"),
    model: Optional[str] = Form(default="mistralai/Voxtral-Mini-3B-2507"),
):
    """Transcribe uploaded audio file with concurrency control."""
    async with request_semaphore:
        try:
            content = await file.read()

            files = {"file": (file.filename, content, file.content_type)}
            data = {"model": model}
            if language and language != "auto":
                data["language"] = language

            response = await http_client.post(
                f"{VOXTRAL_VLLM_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
            )

            return JSONResponse(
                status_code=response.status_code,
                content=response.json(),
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        response = await http_client.get(f"{VOXTRAL_VLLM_URL}/health")
        backend_healthy = response.status_code == 200
    except Exception:
        backend_healthy = False

    return {
        "status": "healthy" if backend_healthy else "degraded",
        "model": "mistralai/Voxtral-Mini-3B-2507",
        "license": "Apache 2.0",
        "backend_healthy": backend_healthy,
        "languages": list(SUPPORTED_LANGUAGES.keys()),
        "optimizations": {
            "vad_enabled": True,
            "chunk_duration_seconds": CHUNK_DURATION_SECONDS,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        },
    }


@app.get("/api/languages")
async def get_languages():
    """Get supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/api/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "active_requests": MAX_CONCURRENT_REQUESTS - request_semaphore._value,
        "chunk_duration": CHUNK_DURATION_SECONDS,
        "vad_threshold": VAD_ENERGY_THRESHOLD,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Voxtral Streaming ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=4302, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
