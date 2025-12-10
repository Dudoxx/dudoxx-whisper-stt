#!/usr/bin/env python3
"""
Advanced Text Processing for Voxtral ASR

Features:
- Improved punctuation restoration (fullstop-multilang-large)
- Named Entity Recognition with GLiNER (names, dates, numbers)
- Semantic paragraph detection with sentence embeddings
- Multi-language support (Arabic, German, French, English, etc.)

License: Apache 2.0
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global model instances (lazy-loaded)
_punctuation_model = None
_ner_model = None
_embedding_model = None


# =============================================================================
# Punctuation Restoration - Upgraded Model
# =============================================================================

def get_punctuation_model():
    """
    Lazy-load the punctuation restoration model.
    Uses fullstop-punctuation-multilang-large for better accuracy.
    """
    global _punctuation_model
    if _punctuation_model is None:
        try:
            from transformers import pipeline

            # Use the larger, more accurate model
            _punctuation_model = pipeline(
                "token-classification",
                model="oliverguhr/fullstop-punctuation-multilang-large",
                aggregation_strategy="simple",
                device="cuda:0",  # Use GPU
            )
            logger.info("Punctuation model loaded: fullstop-punctuation-multilang-large (GPU)")
        except Exception as e:
            logger.warning(f"Failed to load fullstop model, falling back: {e}")
            try:
                # Fallback to original model
                from deepmultilingualpunctuation import PunctuationModel
                _punctuation_model = PunctuationModel()
                logger.info("Punctuation model loaded: deepmultilingualpunctuation (fallback)")
            except Exception as e2:
                logger.warning(f"Failed to load punctuation model: {e2}")
                _punctuation_model = False
    return _punctuation_model if _punctuation_model else None


class PunctuationRestorer:
    """
    Restore punctuation in ASR transcripts.
    Supports: English, German, French, Italian
    """

    def __init__(self):
        self.model = get_punctuation_model()
        self.is_pipeline = hasattr(self.model, '__call__') and not hasattr(self.model, 'restore_punctuation')

    def restore(self, text: str) -> str:
        """Restore punctuation to raw text."""
        if not text or not text.strip():
            return text

        if self.model is None:
            return text.capitalize()

        try:
            if self.is_pipeline:
                # Using transformers pipeline (fullstop model)
                return self._restore_with_pipeline(text)
            else:
                # Using deepmultilingualpunctuation
                return self.model.restore_punctuation(text)
        except Exception as e:
            logger.warning(f"Punctuation restoration failed: {e}")
            return text.capitalize()

    def _restore_with_pipeline(self, text: str) -> str:
        """Restore punctuation using transformers pipeline."""
        # The model predicts punctuation after each word
        results = self.model(text)

        words = text.split()
        punctuated_words = []

        for i, word in enumerate(words):
            punctuated_words.append(word)

            # Find prediction for this word position
            for pred in results:
                if pred['word'].strip() == word or word in pred['word']:
                    label = pred['entity_group']
                    if label == '0':  # Period
                        punctuated_words[-1] += '.'
                    elif label == ',':  # Comma
                        punctuated_words[-1] += ','
                    elif label == '?':  # Question
                        punctuated_words[-1] += '?'
                    elif label == '-':  # Dash/colon
                        punctuated_words[-1] += ':'
                    break

        result = ' '.join(punctuated_words)

        # Capitalize first letter and after sentence endings
        result = self._capitalize_sentences(result)

        return result

    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of sentences."""
        if not text:
            return text

        # Capitalize first character
        result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Capitalize after . ! ?
        result = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)

        return result


# =============================================================================
# Named Entity Recognition - GLiNER
# =============================================================================

def get_ner_model():
    """
    Lazy-load GLiNER model for multilingual NER.
    Supports: Arabic, German, French, English + 16 more languages
    """
    global _ner_model
    if _ner_model is None:
        try:
            from gliner import GLiNER

            # Use the multilingual model
            _ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                _ner_model = _ner_model.to("cuda")
                logger.info("GLiNER NER model loaded on GPU")
            else:
                logger.info("GLiNER NER model loaded on CPU")

        except Exception as e:
            logger.warning(f"Failed to load GLiNER model: {e}")
            _ner_model = False
    return _ner_model if _ner_model else None


@dataclass
class Entity:
    """Named entity with position and type."""
    text: str
    label: str  # PERSON, DATE, NUMBER, ORGANIZATION, LOCATION, etc.
    start: int  # Character position
    end: int
    confidence: float = 0.0


class NamedEntityRecognizer:
    """
    Extract named entities from text using GLiNER.

    Supports custom entity types including:
    - Person names (Arabic, German, French, etc.)
    - Dates (various formats)
    - Numbers/amounts
    - Organizations
    - Locations
    """

    # Entity labels to extract - GLiNER works better with descriptive labels
    DEFAULT_LABELS = [
        # Person names
        "person name",
        "full name",
        # Locations - be specific
        "country",
        "city",
        "country name",
        "city name",
        "geographic location",
        # Organizations
        "company",
        "organization name",
        "institution",
        # Date/time
        "date",
        "time",
        # Numbers
        "number",
        "amount of money",
        "phone number",
        "email address",
        "percentage",
    ]

    # Arabic name patterns (common prefixes/patterns)
    ARABIC_NAME_PATTERNS = [
        r'\b(محمد|أحمد|علي|حسن|حسين|عبد|ابن|أبو|آل)\b',
        r'\b(فاطمة|عائشة|مريم|زينب|خديجة)\b',
    ]

    # German name patterns
    GERMAN_NAME_PATTERNS = [
        r'\b(Hans|Klaus|Peter|Wolfgang|Friedrich|Heinrich|Werner|Karl)\b',
        r'\b(Müller|Schmidt|Schneider|Fischer|Weber|Meyer|Wagner|Becker)\b',
    ]

    # French name patterns
    FRENCH_NAME_PATTERNS = [
        r'\b(Jean|Pierre|Jacques|Michel|François|André|Philippe|Louis)\b',
        r'\b(Marie|Jeanne|Françoise|Monique|Catherine|Nathalie|Isabelle)\b',
        r'\b(Dupont|Martin|Bernard|Dubois|Thomas|Robert|Richard|Petit)\b',
    ]

    # Label normalization map (GLiNER label -> display label)
    LABEL_MAP = {
        "person name": "PERSON",
        "full name": "PERSON",
        "country": "LOCATION",
        "city": "LOCATION",
        "country name": "LOCATION",
        "city name": "LOCATION",
        "geographic location": "LOCATION",
        "company": "ORGANIZATION",
        "organization name": "ORGANIZATION",
        "institution": "ORGANIZATION",
        "date": "DATE",
        "time": "TIME",
        "number": "NUMBER",
        "amount of money": "MONEY",
        "phone number": "PHONE",
        "email address": "EMAIL",
        "percentage": "PERCENTAGE",
    }

    def __init__(self, labels: list[str] | None = None, threshold: float = 0.25):
        self.model = get_ner_model()
        self.labels = labels or self.DEFAULT_LABELS
        self.threshold = threshold

    def extract(self, text: str) -> list[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of Entity objects with text, label, position, confidence
        """
        if not text or not text.strip():
            return []

        entities = []

        # GLiNER extraction
        if self.model is not None:
            try:
                predictions = self.model.predict_entities(
                    text,
                    self.labels,
                    threshold=self.threshold,
                )

                for pred in predictions:
                    # Normalize label using the label map
                    raw_label = pred["label"].lower()
                    normalized_label = self.LABEL_MAP.get(raw_label, raw_label.upper())
                    entities.append(Entity(
                        text=pred["text"],
                        label=normalized_label,
                        start=pred["start"],
                        end=pred["end"],
                        confidence=pred.get("score", 0.0),
                    ))
            except Exception as e:
                logger.warning(f"GLiNER extraction failed: {e}")

        # Supplement with regex patterns for names
        entities.extend(self._extract_with_patterns(text))

        # Extract dates and numbers with regex (backup)
        entities.extend(self._extract_dates_numbers(text))

        # Deduplicate (prefer GLiNER results)
        entities = self._deduplicate(entities)

        return entities

    def _extract_with_patterns(self, text: str) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        all_patterns = (
            self.ARABIC_NAME_PATTERNS +
            self.GERMAN_NAME_PATTERNS +
            self.FRENCH_NAME_PATTERNS
        )

        for pattern in all_patterns:
            for match in re.finditer(pattern, text, re.UNICODE):
                entities.append(Entity(
                    text=match.group(),
                    label="PERSON",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,  # Pattern-based confidence
                ))

        return entities

    def _extract_dates_numbers(self, text: str) -> list[Entity]:
        """Extract dates and numbers using regex."""
        entities = []

        # Date patterns (various formats)
        date_patterns = [
            # ISO format: 2024-01-15
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            # European: 15/01/2024, 15.01.2024
            r'\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b',
            # Written: January 15, 2024 / 15 January 2024
            r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,]?\s+\d{2,4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}[,]?\s+\d{2,4})\b',
            # French: 15 janvier 2024
            r'\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4})\b',
            # German: 15. Januar 2024
            r'\b(\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{2,4})\b',
            # Arabic months (transliterated common patterns)
            r'\b(\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{2,4})\b',
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.UNICODE):
                entities.append(Entity(
                    text=match.group(),
                    label="DATE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))

        # Number patterns
        number_patterns = [
            # Currency: $100, €50, 100€, 50$
            r'([$€£¥]\s*[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*[$€£¥])',
            # Large numbers with separators: 1,000,000 or 1.000.000
            r'\b(\d{1,3}(?:[,.\s]\d{3})+(?:[.,]\d{2})?)\b',
            # Percentages: 50%, 3.5%
            r'\b(\d+(?:[.,]\d+)?\s*%)\b',
            # Phone numbers (various formats)
            r'(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})',
        ]

        for pattern in number_patterns:
            for match in re.finditer(pattern, text):
                label = "NUMBER"
                if '$' in match.group() or '€' in match.group() or '£' in match.group():
                    label = "MONEY"
                elif '%' in match.group():
                    label = "PERCENTAGE"
                elif len(re.findall(r'\d', match.group())) >= 7:
                    label = "PHONE NUMBER"

                entities.append(Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                ))

        return entities

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities, preferring higher confidence."""
        if not entities:
            return []

        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: -e.confidence)

        result = []
        seen_spans = set()

        for entity in sorted_entities:
            span = (entity.start, entity.end)
            # Check for overlap
            is_overlap = False
            for seen_start, seen_end in seen_spans:
                if not (entity.end <= seen_start or entity.start >= seen_end):
                    is_overlap = True
                    break

            if not is_overlap:
                result.append(entity)
                seen_spans.add(span)

        # Sort by position
        return sorted(result, key=lambda e: e.start)


# =============================================================================
# Semantic Paragraph Detection
# =============================================================================

def get_embedding_model():
    """
    Lazy-load sentence embedding model for semantic analysis.
    Uses MiniLM for fast, accurate embeddings.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            # Fast multilingual model
            _embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device='cuda'
            )
            logger.info("Sentence embedding model loaded on GPU")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            _embedding_model = False
    return _embedding_model if _embedding_model else None


@dataclass
class Paragraph:
    """Paragraph with sentences and metadata."""
    index: int
    sentences: list[str]
    text: str
    start_time: float
    end_time: float
    speaker: str | None = None
    topic_shift: bool = False


class SemanticParagraphDetector:
    """
    Detect paragraph breaks using multiple signals:
    - Long pauses (>2s)
    - Speaker changes
    - Topic shifts (semantic similarity)
    - Question-answer patterns
    """

    def __init__(
        self,
        pause_threshold: float = 2.0,
        similarity_threshold: float = 0.5,  # Below this = topic shift
    ):
        self.pause_threshold = pause_threshold
        self.similarity_threshold = similarity_threshold
        self.embedding_model = get_embedding_model()

        self.paragraphs: list[Paragraph] = []
        self.current_sentences: list[dict] = []  # {text, start, end, speaker, embedding}
        self.last_end_time = 0.0
        self.last_speaker: str | None = None

    def add_sentence(
        self,
        text: str,
        start_time: float,
        end_time: float,
        speaker: str | None = None,
    ) -> bool:
        """
        Add a sentence and detect if paragraph break needed.

        Returns:
            True if a paragraph break was detected
        """
        if not text.strip():
            return False

        paragraph_break = False
        break_reason = None

        # Signal 1: Long pause
        if self.current_sentences and start_time - self.last_end_time > self.pause_threshold:
            paragraph_break = True
            break_reason = "pause"

        # Signal 2: Speaker change
        if self.last_speaker and speaker and speaker != self.last_speaker:
            paragraph_break = True
            break_reason = "speaker_change"

        # Signal 3: Topic shift (semantic)
        if self.embedding_model and self.current_sentences and not paragraph_break:
            try:
                # Get embedding for new sentence
                new_embedding = self.embedding_model.encode(text, convert_to_numpy=True)

                # Compare with recent context (last 2-3 sentences)
                recent_texts = [s['text'] for s in self.current_sentences[-3:]]
                if recent_texts:
                    context_embedding = self.embedding_model.encode(
                        ' '.join(recent_texts), convert_to_numpy=True
                    )

                    # Cosine similarity
                    similarity = np.dot(new_embedding, context_embedding) / (
                        np.linalg.norm(new_embedding) * np.linalg.norm(context_embedding) + 1e-8
                    )

                    if similarity < self.similarity_threshold:
                        paragraph_break = True
                        break_reason = "topic_shift"
            except Exception as e:
                logger.debug(f"Embedding comparison failed: {e}")

        # Signal 4: Question followed by statement (new paragraph for answer)
        if self.current_sentences:
            last_text = self.current_sentences[-1]['text'].strip()
            if last_text.endswith('?') and not text.strip().endswith('?'):
                # Question followed by non-question = likely answer = new paragraph
                paragraph_break = True
                break_reason = "qa_pattern"

        # Create new paragraph if break detected
        if paragraph_break and self.current_sentences:
            self._finalize_paragraph(topic_shift=(break_reason == "topic_shift"))

        # Add sentence to current paragraph
        sentence_data = {
            'text': text,
            'start': start_time,
            'end': end_time,
            'speaker': speaker,
        }
        self.current_sentences.append(sentence_data)

        self.last_end_time = end_time
        self.last_speaker = speaker

        return paragraph_break

    def _finalize_paragraph(self, topic_shift: bool = False):
        """Finalize current paragraph."""
        if not self.current_sentences:
            return

        sentences = [s['text'] for s in self.current_sentences]
        speakers = [s['speaker'] for s in self.current_sentences if s['speaker']]

        paragraph = Paragraph(
            index=len(self.paragraphs),
            sentences=sentences,
            text=' '.join(sentences),
            start_time=self.current_sentences[0]['start'],
            end_time=self.current_sentences[-1]['end'],
            speaker=speakers[0] if speakers else None,
            topic_shift=topic_shift,
        )

        self.paragraphs.append(paragraph)
        self.current_sentences = []

    def finalize(self) -> list[Paragraph]:
        """Finalize any remaining sentences as last paragraph."""
        self._finalize_paragraph()
        return self.paragraphs

    def get_paragraphs(self) -> list[dict]:
        """Get paragraphs as dicts for JSON serialization."""
        return [
            {
                'index': p.index,
                'text': p.text,
                'sentence_count': len(p.sentences),
                'start': round(p.start_time, 3),
                'end': round(p.end_time, 3),
                'speaker': p.speaker,
                'topic_shift': p.topic_shift,
            }
            for p in self.paragraphs
        ]

    def reset(self):
        """Reset for new session."""
        self.paragraphs = []
        self.current_sentences = []
        self.last_end_time = 0.0
        self.last_speaker = None


# =============================================================================
# Combined Text Processor
# =============================================================================

@dataclass
class ProcessedText:
    """Result from text processing."""
    # Original and processed text
    raw_text: str
    text: str  # Punctuated

    # Sentences
    sentences: list[dict] = field(default_factory=list)

    # Named entities
    entities: list[dict] = field(default_factory=list)

    # Structure
    is_question: bool = False
    is_complete_sentence: bool = False
    paragraph_break: bool = False

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Processing times
    punctuation_ms: float = 0.0
    ner_ms: float = 0.0
    paragraph_ms: float = 0.0


class TextProcessor:
    """
    Combined text processing with all enhancements:
    - Punctuation restoration
    - Named entity recognition
    - Semantic paragraph detection
    - Word timestamp estimation
    """

    def __init__(
        self,
        enable_punctuation: bool = True,
        enable_ner: bool = True,
        enable_paragraphs: bool = True,
        ner_labels: list[str] | None = None,
    ):
        self.enable_punctuation = enable_punctuation
        self.enable_ner = enable_ner
        self.enable_paragraphs = enable_paragraphs

        # Initialize components
        self.punctuator = PunctuationRestorer() if enable_punctuation else None
        self.ner = NamedEntityRecognizer(labels=ner_labels) if enable_ner else None
        self.paragraph_detector = SemanticParagraphDetector() if enable_paragraphs else None

        # Session state
        self.sentences: list[dict] = []
        self.all_entities: list[dict] = []

    def process(
        self,
        raw_text: str,
        start_time: float,
        end_time: float,
        speaker: str | None = None,
        is_final: bool = False,
    ) -> ProcessedText:
        """
        Process raw ASR text with all enhancements.

        Args:
            raw_text: Raw transcription from ASR
            start_time: Start timestamp
            end_time: End timestamp
            speaker: Current speaker label
            is_final: Whether this is a final (not partial) result

        Returns:
            ProcessedText with all enhancements
        """
        result = ProcessedText(
            raw_text=raw_text,
            text=raw_text,
            start_time=start_time,
            end_time=end_time,
        )

        if not raw_text or not raw_text.strip():
            return result

        # 1. Punctuation restoration
        if self.punctuator:
            t0 = time.time()
            result.text = self.punctuator.restore(raw_text)
            result.punctuation_ms = (time.time() - t0) * 1000

        # 2. Named entity recognition
        if self.ner:
            t0 = time.time()
            entities = self.ner.extract(result.text)
            result.entities = [
                {
                    'text': e.text,
                    'label': e.label,
                    'start': e.start,
                    'end': e.end,
                    'confidence': round(e.confidence, 2),
                }
                for e in entities
            ]
            result.ner_ms = (time.time() - t0) * 1000

            # Track all entities in session
            self.all_entities.extend(result.entities)

        # 3. Sentence detection
        sentences = self._detect_sentences(result.text, start_time, end_time)
        result.sentences = sentences

        # 4. Paragraph detection
        if self.paragraph_detector and sentences:
            t0 = time.time()
            for sent in sentences:
                result.paragraph_break = self.paragraph_detector.add_sentence(
                    sent['text'],
                    sent['start'],
                    sent['end'],
                    speaker,
                )
            result.paragraph_ms = (time.time() - t0) * 1000

        # 5. Structure detection
        result.is_question = result.text.rstrip().endswith('?')
        result.is_complete_sentence = any(
            result.text.rstrip().endswith(p) for p in '.!?'
        )

        # Track sentences
        self.sentences.extend(sentences)

        return result

    def _detect_sentences(
        self,
        text: str,
        start_time: float,
        end_time: float,
    ) -> list[dict]:
        """Split text into sentences with timestamp estimates."""
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
                'index': len(self.sentences) + i,
                'text': part,
                'start': round(current_time, 3),
                'end': round(current_time + part_duration, 3),
                'word_count': len(part.split()),
            })

            current_time += part_duration

        return sentences

    def get_paragraphs(self) -> list[dict]:
        """Get all detected paragraphs."""
        if self.paragraph_detector:
            return self.paragraph_detector.get_paragraphs()
        return []

    def get_entities_summary(self) -> dict:
        """Get summary of all detected entities by type."""
        summary = {}
        for entity in self.all_entities:
            label = entity['label']
            if label not in summary:
                summary[label] = []
            if entity['text'] not in summary[label]:
                summary[label].append(entity['text'])
        return summary

    def reset(self):
        """Reset for new session."""
        self.sentences = []
        self.all_entities = []
        if self.paragraph_detector:
            self.paragraph_detector.reset()


# =============================================================================
# Preload Models
# =============================================================================

def preload_text_models(
    enable_punctuation: bool = True,
    enable_ner: bool = True,
    enable_embeddings: bool = True,
):
    """Preload text processing models at startup."""
    if enable_punctuation:
        logger.info("Preloading punctuation model...")
        model = get_punctuation_model()
        if model:
            logger.info("Punctuation model loaded successfully")
        else:
            logger.warning("Punctuation model failed to load")

    if enable_ner:
        logger.info("Preloading GLiNER NER model...")
        model = get_ner_model()
        if model:
            logger.info("GLiNER NER model loaded successfully")
        else:
            logger.warning("GLiNER NER model failed to load")

    if enable_embeddings:
        logger.info("Preloading sentence embedding model...")
        model = get_embedding_model()
        if model:
            logger.info("Sentence embedding model loaded successfully")
        else:
            logger.warning("Sentence embedding model failed to load")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the processor
    processor = TextProcessor()

    test_texts = [
        "hello my name is mohammed ali and i live in berlin",
        "the meeting is scheduled for january 15 2024 at 3 pm",
        "please contact hans müller at the office",
        "le rendez vous est prévu pour le 20 mars",
        "the total cost is 1500 euros including tax",
    ]

    for text in test_texts:
        result = processor.process(text, 0.0, 3.0)
        print(f"\nInput: {text}")
        print(f"Output: {result.text}")
        print(f"Entities: {result.entities}")
        print(f"Timing: punct={result.punctuation_ms:.1f}ms, ner={result.ner_ms:.1f}ms")
