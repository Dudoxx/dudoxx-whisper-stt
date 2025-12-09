// Recording types
export interface Recording {
  id: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  duration: number; // in seconds
  audioBlob?: Blob;
  audioUrl?: string;
  transcriptionId?: string;
  language?: string;
  metadata?: RecordingMetadata;
}

export interface RecordingMetadata {
  sampleRate: number;
  channels: number;
  mimeType: string;
  deviceId?: string;
  deviceLabel?: string;
}

// Transcription types
export interface Transcription {
  id: string;
  recordingId: string;
  createdAt: Date;
  updatedAt: Date;
  language: string;
  segments: TranscriptionSegment[];
  fullText: string;
  speakers: Speaker[];
  metadata?: TranscriptionMetadata;
}

export interface TranscriptionSegment {
  id: string;
  start: number; // in seconds
  end: number; // in seconds
  text: string;
  speaker?: string;
  confidence?: number;
  words?: TranscriptionWord[];
}

export interface TranscriptionWord {
  text: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface Speaker {
  id: string;
  label: string;
  color?: string;
}

export interface TranscriptionMetadata {
  model: string;
  backend: string;
  processingTime?: number;
  wordCount: number;
  segmentCount: number;
}

// WebSocket message types
export type WSMessageType =
  | "transcript"
  | "partial"
  | "final"
  | "error"
  | "status"
  | "config"
  | "speaker"
  | "speaker";

export interface WSMessage {
  type: WSMessageType;
  data: WSTranscriptData | WSStatusData | WSErrorData;
  timestamp: number;
}

export interface WSTranscriptData {
  text: string;
  start?: number;
  end?: number;
  speaker?: string;
  confidence?: number;
  isFinal: boolean;
  words?: TranscriptionWord[];
}

export interface WSStatusData {
  status: "connected" | "disconnected" | "processing" | "ready";
  message?: string;
}

export interface WSErrorData {
  code: string;
  message: string;
}

// Recording state
export type RecordingStatus =
  | "idle"
  | "recording"
  | "paused"
  | "processing"
  | "error";

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

export interface RecordingState {
  status: RecordingStatus;
  connectionStatus: ConnectionStatus;
  duration: number;
  currentTranscript: string;
  segments: TranscriptionSegment[];
  error?: string;
}

// Settings types
export interface AppSettings {
  serverUrl: string;
  language: string;
  theme: "light" | "dark" | "system";
  autoSave: boolean;
  showTimestamps: boolean;
  showSpeakers: boolean;
}

// Export types
export type ExportFormat = "txt" | "json" | "srt" | "vtt";

export interface ExportOptions {
  format: ExportFormat;
  includeTimestamps: boolean;
  includeSpeakers: boolean;
  filename?: string;
}

// Storage types
export interface StorageInfo {
  used: number;
  available: number;
  recordings: number;
  transcriptions: number;
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

// Locale type
export type Locale = "en" | "fr" | "de";
