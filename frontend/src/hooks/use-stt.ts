"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { STTClient } from "@/lib/websocket";
import { useAudioRecorder } from "./use-audio-recorder";
import type {
  ConnectionStatus,
  RecordingStatus,
  TranscriptionSegment,
  WSTranscriptData,
  WSErrorData,
} from "@/types";

export interface UseSTTOptions {
  serverUrl: string;
  autoConnect?: boolean;
}

export interface UseSTTState {
  connectionStatus: ConnectionStatus;
  recordingStatus: RecordingStatus;
  duration: number;
  audioLevel: number;
  currentTranscript: string;
  segments: TranscriptionSegment[];
  error: string | null;
}

export interface StopRecordingResult {
  blob: Blob | null;
  segments: TranscriptionSegment[];
}

export interface UseSTTControls {
  connect: () => void;
  disconnect: () => void;
  startRecording: (deviceId?: string) => Promise<void>;
  stopRecording: () => Promise<StopRecordingResult>;
  pauseRecording: () => void;
  resumeRecording: () => void;
  clearTranscript: () => void;
  getDevices: () => Promise<MediaDeviceInfo[]>;
}

export function useSTT(options: UseSTTOptions): [UseSTTState, UseSTTControls] {
  const { serverUrl, autoConnect = false } = options;

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [currentTranscript, setCurrentTranscript] = useState("");
  const [segments, setSegments] = useState<TranscriptionSegment[]>([]);
  const [error, setError] = useState<string | null>(null);

  const clientRef = useRef<STTClient | null>(null);

  // Audio recorder - sends WebM chunks to server
  const [recorderState, recorderControls] = useAudioRecorder({
    channelCount: 1,
    chunkDuration: 100,
    onAudioData: (blob: Blob) => {
      if (clientRef.current?.isConnected()) {
        clientRef.current.sendAudio(blob);
      }
    },
    onError: (err) => {
      setError(err.message);
    },
  });

  const recordingStatus: RecordingStatus = recorderState.isPaused
    ? "paused"
    : recorderState.isRecording
      ? "recording"
      : "idle";

  // Handle transcript updates (buffer/interim text)
  const handleTranscript = useCallback((data: WSTranscriptData) => {
    if (data.isFinal) {
      setCurrentTranscript("");
    } else {
      setCurrentTranscript(data.text);
    }
  }, []);

  // Handle new finalized segments
  const handleSegment = useCallback((segment: TranscriptionSegment) => {
    setSegments((prev) => [...prev, segment]);
  }, []);

  // Handle connection status
  const handleStatus = useCallback((status: ConnectionStatus, message?: string) => {
    setConnectionStatus(status);
    if (status === "error" && message) {
      setError(message);
    }
  }, []);

  // Handle errors
  const handleError = useCallback((err: WSErrorData) => {
    setError(err.message);
  }, []);

  // Initialize client once
  useEffect(() => {
    clientRef.current = new STTClient({
      serverUrl,
      onTranscript: handleTranscript,
      onSegment: handleSegment,
      onStatus: handleStatus,
      onError: handleError,
    });

    if (autoConnect) {
      clientRef.current.connect();
    }

    return () => {
      clientRef.current?.disconnect();
    };
  }, [serverUrl, autoConnect, handleTranscript, handleSegment, handleStatus, handleError]);

  const connect = useCallback(() => {
    setError(null);
    clientRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    clientRef.current?.disconnect();
  }, []);

  const startRecording = useCallback(
    async (deviceId?: string) => {
      setError(null);
      setCurrentTranscript("");
      setSegments([]);
      clientRef.current?.reset();

      // Connect if not connected
      if (!clientRef.current?.isConnected()) {
        clientRef.current?.connect();

        // Wait for connection
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error("Connection timeout")), 5000);
          const check = setInterval(() => {
            if (clientRef.current?.isConnected()) {
              clearTimeout(timeout);
              clearInterval(check);
              resolve();
            }
          }, 100);
        });
      }

      // Start recording (acquires mic)
      await recorderControls.start(deviceId);
    },
    [recorderControls],
  );

  const stopRecording = useCallback(async (): Promise<StopRecordingResult> => {
    // Stop recording (releases mic, returns blob)
    const blob = await recorderControls.stop();
    
    // Wait a moment for final messages, then finalize segments
    await new Promise(resolve => setTimeout(resolve, 300));
    const finalSegments = clientRef.current?.finalizeSegments() || [];
    
    // Update segments state with finalized segments
    setSegments(finalSegments);
    
    // Clear the interim transcript
    setCurrentTranscript("");
    
    return { blob, segments: finalSegments };
  }, [recorderControls]);

  const pauseRecording = useCallback(() => {
    recorderControls.pause();
  }, [recorderControls]);

  const resumeRecording = useCallback(() => {
    recorderControls.resume();
  }, [recorderControls]);

  const clearTranscript = useCallback(() => {
    setCurrentTranscript("");
    setSegments([]);
    clientRef.current?.reset();
  }, []);

  const getDevices = useCallback(async () => {
    return recorderControls.getDevices();
  }, [recorderControls]);

  return [
    {
      connectionStatus,
      recordingStatus,
      duration: recorderState.duration,
      audioLevel: recorderState.audioLevel,
      currentTranscript,
      segments,
      error,
    },
    {
      connect,
      disconnect,
      startRecording,
      stopRecording,
      pauseRecording,
      resumeRecording,
      clearTranscript,
      getDevices,
    },
  ];
}
