"use client";

import { useState, useRef, useCallback, useEffect } from "react";

export interface AudioRecorderOptions {
  channelCount?: number;
  chunkDuration?: number; // ms between chunks sent to server
  onAudioData?: (data: Blob) => void;
  onError?: (error: Error) => void;
}

export interface AudioRecorderState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioLevel: number;
  deviceId: string | null;
  deviceLabel: string | null;
}

export interface AudioRecorderControls {
  start: (deviceId?: string) => Promise<void>;
  stop: () => Promise<Blob | null>;
  pause: () => void;
  resume: () => void;
  getDevices: () => Promise<MediaDeviceInfo[]>;
}

const DEFAULT_CHANNEL_COUNT = 1;

export function useAudioRecorder(
  options: AudioRecorderOptions = {},
): [AudioRecorderState, AudioRecorderControls] {
  const {
    channelCount = DEFAULT_CHANNEL_COUNT,
    chunkDuration = 100,
    onAudioData,
    onError,
  } = options;

  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    audioLevel: 0,
    deviceId: null,
    deviceLabel: null,
  });

  // Refs for callback access
  const isPausedRef = useRef(false);
  const isRecordingRef = useRef(false);
  const onAudioDataRef = useRef(onAudioData);

  useEffect(() => {
    onAudioDataRef.current = onAudioData;
  }, [onAudioData]);

  // Media refs
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamingRecorderRef = useRef<MediaRecorder | null>(null);
  const saveRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  
  // Timer refs
  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const levelIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const pausedAtRef = useRef<number>(0);
  const totalPausedRef = useRef<number>(0);

  // Full cleanup - releases microphone
  const cleanup = useCallback(() => {
    console.log("[AudioRecorder] Cleanup - releasing all resources");
    isRecordingRef.current = false;
    isPausedRef.current = false;

    // Clear timers
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    if (levelIntervalRef.current) {
      clearInterval(levelIntervalRef.current);
      levelIntervalRef.current = null;
    }

    // Stop streaming recorder
    if (streamingRecorderRef.current && streamingRecorderRef.current.state !== "inactive") {
      try {
        streamingRecorderRef.current.stop();
      } catch {}
    }
    streamingRecorderRef.current = null;

    // Stop save recorder
    if (saveRecorderRef.current && saveRecorderRef.current.state !== "inactive") {
      try {
        saveRecorderRef.current.stop();
      } catch {}
    }
    saveRecorderRef.current = null;

    // Disconnect analyser
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch {}
      analyserRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    // CRITICAL: Stop all tracks to release microphone
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => {
        console.log("[AudioRecorder] Stopping track:", track.kind, track.label);
        track.stop();
      });
      mediaStreamRef.current = null;
    }

    chunksRef.current = [];
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Get available microphones (does NOT acquire mic)
  const getDevices = useCallback(async (): Promise<MediaDeviceInfo[]> => {
    try {
      // Check if we have permission without acquiring
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter((d) => d.kind === "audioinput");
      
      // If no labels, we need to request permission temporarily
      if (audioInputs.length > 0 && !audioInputs[0].label) {
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const devicesWithLabels = await navigator.mediaDevices.enumerateDevices();
        // Immediately release the temp stream
        tempStream.getTracks().forEach((track) => track.stop());
        return devicesWithLabels.filter((d) => d.kind === "audioinput");
      }
      
      return audioInputs;
    } catch (error) {
      onError?.(error instanceof Error ? error : new Error("Failed to get devices"));
      return [];
    }
  }, [onError]);

  // Start recording - acquires microphone HERE
  const start = useCallback(
    async (deviceId?: string): Promise<void> => {
      try {
        // Ensure clean state
        cleanup();
        isPausedRef.current = false;
        totalPausedRef.current = 0;

        // Acquire microphone
        const constraints: MediaStreamConstraints = {
          audio: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            channelCount: { exact: channelCount },
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        };

        console.log("[AudioRecorder] Acquiring microphone...");
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        mediaStreamRef.current = stream;
        console.log("[AudioRecorder] Microphone acquired");

        const audioTrack = stream.getAudioTracks()[0];
        const settings = audioTrack.getSettings();

        // Setup AudioContext for level visualization
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }

        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        source.connect(analyser);

        // Determine supported mime type
        let mimeType = "audio/webm;codecs=opus";
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = "audio/webm";
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = "audio/mp4";
            if (!MediaRecorder.isTypeSupported(mimeType)) {
              mimeType = "";
            }
          }
        }
        console.log("[AudioRecorder] Using mimeType:", mimeType || "default");

        // Create streaming recorder (sends chunks to server)
        const streamingRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        streamingRecorderRef.current = streamingRecorder;

        streamingRecorder.ondataavailable = (event) => {
          if (event.data.size > 0 && !isPausedRef.current && isRecordingRef.current) {
            onAudioDataRef.current?.(event.data);
          }
        };

        // Create save recorder (collects full audio)
        const saveRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        saveRecorderRef.current = saveRecorder;

        saveRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        };

        // Start both recorders
        streamingRecorder.start(chunkDuration);
        saveRecorder.start(1000);

        // Start timing
        startTimeRef.current = Date.now();
        isRecordingRef.current = true;

        // Duration timer
        durationIntervalRef.current = setInterval(() => {
          if (isRecordingRef.current && !isPausedRef.current) {
            const elapsed = Date.now() - startTimeRef.current - totalPausedRef.current;
            setState((prev) => ({ ...prev, duration: Math.floor(elapsed / 1000) }));
          }
        }, 100);

        // Audio level timer
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        levelIntervalRef.current = setInterval(() => {
          if (analyserRef.current && isRecordingRef.current) {
            analyserRef.current.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
            setState((prev) => ({ ...prev, audioLevel: average / 255 }));
          }
        }, 50);

        setState({
          isRecording: true,
          isPaused: false,
          duration: 0,
          audioLevel: 0,
          deviceId: settings.deviceId || null,
          deviceLabel: audioTrack.label || null,
        });

        console.log("[AudioRecorder] Recording started");
      } catch (error) {
        console.error("[AudioRecorder] Failed to start:", error);
        cleanup();
        onError?.(error instanceof Error ? error : new Error("Failed to start recording"));
      }
    },
    [chunkDuration, channelCount, onError, cleanup],
  );

  // Stop recording - returns audio blob and releases microphone
  const stop = useCallback(async (): Promise<Blob | null> => {
    console.log("[AudioRecorder] Stop called");
    isRecordingRef.current = false;

    return new Promise((resolve) => {
      const saveRecorder = saveRecorderRef.current;
      
      if (!saveRecorder || saveRecorder.state === "inactive") {
        cleanup();
        setState((prev) => ({
          ...prev,
          isRecording: false,
          isPaused: false,
          audioLevel: 0,
        }));
        resolve(null);
        return;
      }

      saveRecorder.onstop = () => {
        const mimeType = saveRecorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        console.log("[AudioRecorder] Created blob:", blob.size, "bytes, type:", mimeType);
        
        // Cleanup releases microphone
        cleanup();
        
        setState((prev) => ({
          ...prev,
          isRecording: false,
          isPaused: false,
          audioLevel: 0,
        }));
        
        resolve(blob);
      };

      // Stop streaming recorder first
      if (streamingRecorderRef.current && streamingRecorderRef.current.state !== "inactive") {
        streamingRecorderRef.current.stop();
      }
      
      // Then stop save recorder (triggers onstop)
      saveRecorder.stop();
    });
  }, [cleanup]);

  // Pause recording
  const pause = useCallback(() => {
    if (!isRecordingRef.current || isPausedRef.current) return;

    isPausedRef.current = true;
    pausedAtRef.current = Date.now();

    if (streamingRecorderRef.current?.state === "recording") {
      streamingRecorderRef.current.pause();
    }
    if (saveRecorderRef.current?.state === "recording") {
      saveRecorderRef.current.pause();
    }

    setState((prev) => ({ ...prev, isPaused: true }));
    console.log("[AudioRecorder] Paused");
  }, []);

  // Resume recording
  const resume = useCallback(() => {
    if (!isRecordingRef.current || !isPausedRef.current) return;

    totalPausedRef.current += Date.now() - pausedAtRef.current;
    isPausedRef.current = false;

    if (streamingRecorderRef.current?.state === "paused") {
      streamingRecorderRef.current.resume();
    }
    if (saveRecorderRef.current?.state === "paused") {
      saveRecorderRef.current.resume();
    }

    setState((prev) => ({ ...prev, isPaused: false }));
    console.log("[AudioRecorder] Resumed");
  }, []);

  return [state, { start, stop, pause, resume, getDevices }];
}
