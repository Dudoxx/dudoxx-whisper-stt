"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useTranslations, useLocale } from "next-intl";
import {
  Mic,
  Square,
  Pause,
  Play,
  Wifi,
  WifiOff,
  Settings,
  Volume2,
  SkipBack,
  SkipForward,
  VolumeX,
  FileText,
} from "lucide-react";
import {
  DDXButton,
  DDXCard,
  DDXCardContent,
  DDXCardHeader,
  DDXCardTitle,
  DDXBadge,
} from "@/components/ddx";
import { useSTT } from "@/hooks";
import {
  saveRecording,
  saveTranscription,
  saveAudioBlob,
  getRecording,
  getTranscription,
  generateId,
} from "@/lib/storage";
import type { Recording, Transcription, TranscriptionSegment } from "@/types";

const SERVER_URL = process.env.NEXT_PUBLIC_STT_SERVER_URL || "ws://localhost:4300/asr";

const SPEAKER_COLORS = [
  "bg-blue-500",
  "bg-green-500",
  "bg-purple-500",
  "bg-orange-500",
];

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

function getSpeakerColor(speaker: string | undefined): string {
  if (!speaker) return "bg-gray-500";
  const num = parseInt(speaker.replace(/\D/g, ""), 10) || 0;
  return SPEAKER_COLORS[(num - 1) % SPEAKER_COLORS.length];
}

interface DeviceOption {
  deviceId: string;
  label: string;
}

export default function RecordPage(): React.ReactNode {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Devices
  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [showDeviceSelect, setShowDeviceSelect] = useState(false);

  // Current session
  const [currentRecordingId, setCurrentRecordingId] = useState<string | null>(null);
  const [currentTranscriptionId, setCurrentTranscriptionId] = useState<string | null>(null);
  const [recordingName, setRecordingName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [savedSegments, setSavedSegments] = useState<TranscriptionSegment[]>([]);

  // Playback state
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const [playbackVolume, setPlaybackVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  const [state, controls] = useSTT({
    serverUrl: SERVER_URL,
    autoConnect: false,
  });

  const {
    connectionStatus,
    recordingStatus,
    duration,
    audioLevel,
    currentTranscript,
    error,
  } = state;

  // Load devices on mount
  useEffect(() => {
    controls.getDevices().then((list) => {
      const opts = list.map((d) => ({
        deviceId: d.deviceId,
        label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
      }));
      setDevices(opts);
      if (opts.length > 0 && !selectedDevice) {
        setSelectedDevice(opts[0].deviceId);
      }
    });
  }, [controls, selectedDevice]);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [savedSegments, currentTranscript]);

  // Manage audio URL
  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setAudioUrl(null);
  }, [audioBlob]);

  // START RECORDING
  const handleStartRecording = useCallback(async () => {
    try {
      const recordingId = generateId();
      const transcriptionId = generateId();
      const now = new Date();
      const name = recordingName || `Recording ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;

      // Create recording in IndexedDB
      const recording: Recording = {
        id: recordingId,
        name,
        createdAt: now,
        updatedAt: now,
        duration: 0,
        transcriptionId,
        language: "auto",
        metadata: { sampleRate: 16000, channels: 1, mimeType: "audio/webm" },
      };
      await saveRecording(recording);

      // Create transcription in IndexedDB
      const transcription: Transcription = {
        id: transcriptionId,
        recordingId,
        createdAt: now,
        updatedAt: now,
        language: "auto",
        segments: [],
        fullText: "",
        speakers: [],
        metadata: { model: "large-v3-turbo", backend: "mlx-whisper", wordCount: 0, segmentCount: 0 },
      };
      await saveTranscription(transcription);

      setCurrentRecordingId(recordingId);
      setCurrentTranscriptionId(transcriptionId);
      setSavedSegments([]);
      console.log("[RecordPage] Created recording:", recordingId, "transcription:", transcriptionId);

      await controls.startRecording(selectedDevice || undefined);
    } catch (err) {
      console.error("[RecordPage] Failed to start:", err);
    }
  }, [controls, selectedDevice, recordingName]);

  // STOP RECORDING
  const handleStopRecording = useCallback(async () => {
    if (isSaving) return;
    setIsSaving(true);

    try {
      const { blob, segments: finalSegments } = await controls.stopRecording();
      console.log("[RecordPage] Stopped, blob:", blob?.size, "segments:", finalSegments.length);

      setSavedSegments(finalSegments);

      if (blob && currentRecordingId) {
        setAudioBlob(blob);
        await saveAudioBlob(currentRecordingId, blob);

        const recording = await getRecording(currentRecordingId);
        if (recording) {
          recording.duration = duration;
          recording.updatedAt = new Date();
          recording.metadata = { sampleRate: 16000, channels: 1, mimeType: blob.type || "audio/webm" };
          await saveRecording(recording);
        }

        // Update transcription with final data
        if (currentTranscriptionId) {
          const transcription = await getTranscription(currentTranscriptionId);
          console.log("[RecordPage] Got transcription:", transcription?.id);
          
          if (transcription) {
            const uniqueSpeakers = new Set<string>();
            finalSegments.forEach((s) => { if (s.speaker) uniqueSpeakers.add(s.speaker); });

            const fullText = finalSegments.map((s) => s.text).join(" ");
            transcription.segments = finalSegments;
            transcription.fullText = fullText;
            transcription.updatedAt = new Date();
            transcription.speakers = Array.from(uniqueSpeakers).map((s, i) => ({
              id: s, label: s, color: SPEAKER_COLORS[i % SPEAKER_COLORS.length],
            }));
            transcription.metadata = {
              model: "large-v3-turbo",
              backend: "mlx-whisper",
              wordCount: fullText.split(/\s+/).filter(Boolean).length,
              segmentCount: finalSegments.length,
            };
            await saveTranscription(transcription);
            console.log("[RecordPage] Saved transcription with", finalSegments.length, "segments");
          } else {
            console.error("[RecordPage] Transcription not found:", currentTranscriptionId);
          }
        }

        console.log("[RecordPage] Saved to IndexedDB");
        
        // Navigate to recording detail page
        router.push(`/${locale}/recordings/${currentRecordingId}`);
      }
    } catch (err) {
      console.error("[RecordPage] Failed to stop:", err);
    } finally {
      setIsSaving(false);
    }
  }, [controls, currentRecordingId, currentTranscriptionId, duration, isSaving, locale, router]);

  const handlePauseRecording = useCallback(() => controls.pauseRecording(), [controls]);
  const handleResumeRecording = useCallback(() => controls.resumeRecording(), [controls]);

  // Playback
  const togglePlayback = useCallback(() => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const seekTo = useCallback((time: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(time, playbackDuration));
    setPlaybackTime(audioRef.current.currentTime);
  }, [playbackDuration]);

  const skipBackward = useCallback(() => seekTo(playbackTime - 10), [playbackTime, seekTo]);
  const skipForward = useCallback(() => seekTo(playbackTime + 10), [playbackTime, seekTo]);

  const toggleMute = useCallback(() => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const vol = parseFloat(e.target.value);
    if (audioRef.current) audioRef.current.volume = vol;
    setPlaybackVolume(vol);
    setIsMuted(vol === 0);
  }, []);

  const handleSeekChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    seekTo(parseFloat(e.target.value));
  }, [seekTo]);

  // View recording
  const handleViewRecording = useCallback(() => {
    if (currentRecordingId) {
      router.push(`/${locale}/recordings/${currentRecordingId}`);
    }
  }, [currentRecordingId, locale, router]);

  // New recording
  const handleNewRecording = useCallback(() => {
    setCurrentRecordingId(null);
    setCurrentTranscriptionId(null);
    setAudioBlob(null);
    setAudioUrl(null);
    setPlaybackTime(0);
    setPlaybackDuration(0);
    setIsPlaying(false);
    setRecordingName("");
    setSavedSegments([]);
    controls.clearTranscript();
    controls.disconnect();
  }, [controls]);

  const isRecording = recordingStatus === "recording";
  const isPaused = recordingStatus === "paused";
  const isConnected = connectionStatus === "connected";
  const hasContent = savedSegments.length > 0 || currentTranscript;
  const canPlayback = audioBlob && !isRecording && !isPaused;
  const hasFinishedRecording = audioBlob !== null && !isRecording && !isPaused;

  // Get status text
  const getStatusText = () => {
    if (hasFinishedRecording) return t("recording.status.processing");
    return t(`recording.status.${recordingStatus}`);
  };

  return (
    <div className="flex h-full flex-col gap-6">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={() => audioRef.current && setPlaybackTime(audioRef.current.currentTime)}
          onLoadedMetadata={() => audioRef.current && setPlaybackDuration(audioRef.current.duration)}
          onEnded={() => { setIsPlaying(false); setPlaybackTime(0); }}
          preload="metadata"
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("recording.title")}</h1>
          <p className="text-muted-foreground">{getStatusText()}</p>
        </div>
        <DDXBadge variant={isConnected ? "success" : "secondary"} className="gap-1">
          {isConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
          {t(`recording.status.${connectionStatus}`)}
        </DDXBadge>
      </div>

      {/* Error */}
      {error && (
        <DDXCard className="border-destructive bg-destructive/10">
          <DDXCardContent className="py-3 text-destructive">{error}</DDXCardContent>
        </DDXCard>
      )}

      {/* Main layout */}
      <div className="grid flex-1 gap-6 lg:grid-cols-3">
        {/* Controls */}
        <DDXCard className="lg:col-span-1">
          <DDXCardHeader>
            <DDXCardTitle className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              {t("recording.title")}
            </DDXCardTitle>
          </DDXCardHeader>
          <DDXCardContent className="space-y-6">
            {/* Duration */}
            <div className="text-center">
              <div className="font-mono text-5xl font-bold tabular-nums">
                {formatDuration(isRecording || isPaused ? duration : (playbackDuration || duration))}
              </div>
              <p className="mt-1 text-sm text-muted-foreground">{t("recording.duration")}</p>
            </div>

            {/* Audio Level */}
            {(isRecording || isPaused) && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <Volume2 className="h-3 w-3" />
                    {t("settings.audio")}
                  </span>
                  <span className="font-mono text-xs">{Math.round(audioLevel * 100)}%</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 transition-all duration-75"
                    style={{ width: `${Math.min(audioLevel * 100, 100)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Mic Selection */}
            {!isRecording && !isPaused && !hasFinishedRecording && (
              <>
                <div className="space-y-2">
                  <label className="text-sm font-medium">{t("recording.microphone")}</label>
                  <button
                    onClick={() => setShowDeviceSelect(!showDeviceSelect)}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-left text-sm hover:bg-accent"
                  >
                    {devices.find((d) => d.deviceId === selectedDevice)?.label || t("recording.selectMicrophone")}
                  </button>
                  {showDeviceSelect && (
                    <div className="rounded-md border border-input bg-popover p-1 shadow-lg">
                      {devices.map((device) => (
                        <button
                          key={device.deviceId}
                          onClick={() => { setSelectedDevice(device.deviceId); setShowDeviceSelect(false); }}
                          className="w-full rounded px-2 py-1.5 text-left text-sm hover:bg-accent"
                        >
                          {device.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">{t("common.save")}</label>
                  <input
                    type="text"
                    value={recordingName}
                    onChange={(e) => setRecordingName(e.target.value)}
                    placeholder={t("recording.title")}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </div>
              </>
            )}

            {/* Recording Controls */}
            <div className="space-y-3">
              {!isRecording && !isPaused && !hasFinishedRecording ? (
                <DDXButton onClick={handleStartRecording} variant="recording" size="xl" className="w-full">
                  <Mic className="mr-2 h-5 w-5" />
                  {t("recording.start")}
                </DDXButton>
              ) : isRecording || isPaused ? (
                <div className="flex gap-2">
                  <DDXButton
                    onClick={isPaused ? handleResumeRecording : handlePauseRecording}
                    variant="secondary"
                    size="lg"
                    className="flex-1"
                  >
                    {isPaused ? (
                      <><Play className="mr-2 h-4 w-4" />{t("recording.resume")}</>
                    ) : (
                      <><Pause className="mr-2 h-4 w-4" />{t("recording.pause")}</>
                    )}
                  </DDXButton>
                  <DDXButton
                    onClick={handleStopRecording}
                    variant="destructive"
                    size="lg"
                    className="flex-1"
                    disabled={isSaving}
                  >
                    <Square className="mr-2 h-4 w-4" />
                    {t("recording.stop")}
                  </DDXButton>
                </div>
              ) : null}

              {/* Playback */}
              {canPlayback && (
                <div className="space-y-4 rounded-lg border border-border bg-muted/30 p-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{t("playback.title")}</span>
                    <span className="font-mono text-muted-foreground">
                      {formatDuration(playbackTime)} / {formatDuration(playbackDuration)}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={playbackDuration || 1}
                    step={0.1}
                    value={playbackTime}
                    onChange={handleSeekChange}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
                  />

                  <div className="flex items-center justify-center gap-2">
                    <DDXButton variant="ghost" size="icon" onClick={skipBackward}>
                      <SkipBack className="h-4 w-4" />
                    </DDXButton>
                    <DDXButton variant="default" size="icon-lg" onClick={togglePlayback} className="h-12 w-12 rounded-full">
                      {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
                    </DDXButton>
                    <DDXButton variant="ghost" size="icon" onClick={skipForward}>
                      <SkipForward className="h-4 w-4" />
                    </DDXButton>
                  </div>

                  <div className="flex items-center gap-2">
                    <DDXButton variant="ghost" size="icon-sm" onClick={toggleMute}>
                      {isMuted || playbackVolume === 0 ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                    </DDXButton>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={isMuted ? 0 : playbackVolume}
                      onChange={handleVolumeChange}
                      className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
                    />
                  </div>
                </div>
              )}

              {/* Post-recording actions */}
              {hasFinishedRecording && (
                <div className="flex gap-2 pt-2">
                  <DDXButton onClick={handleViewRecording} variant="success" size="lg" className="flex-1">
                    <FileText className="mr-2 h-4 w-4" />
                    {t("nav.recordings")}
                  </DDXButton>
                  <DDXButton onClick={handleNewRecording} variant="secondary" size="lg" className="flex-1">
                    <Mic className="mr-2 h-4 w-4" />
                    {t("nav.newRecording")}
                  </DDXButton>
                </div>
              )}
            </div>
          </DDXCardContent>
        </DDXCard>

        {/* Transcription Panel */}
        <DDXCard className="flex flex-col lg:col-span-2">
          <DDXCardHeader className="border-b">
            <DDXCardTitle className="flex items-center justify-between">
              <span>{t("transcription.title")}</span>
              {savedSegments.length > 0 && (
                <DDXBadge variant="secondary">
                  {savedSegments.length} {t("transcription.sentences")}
                </DDXBadge>
              )}
            </DDXCardTitle>
          </DDXCardHeader>
          <DDXCardContent className="flex-1 overflow-auto p-0">
            <div className="h-full min-h-[400px] p-4">
              {!hasContent ? (
                <div className="flex h-full flex-col items-center justify-center text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                    <Mic className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="mt-4 text-muted-foreground">{t("transcription.noTranscription")}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{t("recording.status.idle")}</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Finalized segments */}
                  {savedSegments.map((segment) => (
                    <SegmentItem key={segment.id} segment={segment} />
                  ))}

                  {/* Current buffer (interim transcript) */}
                  {currentTranscript && (
                    <div className="rounded-lg border border-dashed border-primary/50 bg-primary/5 p-3">
                      <p className="italic text-foreground/70">{currentTranscript}</p>
                    </div>
                  )}

                  {/* Recording indicator */}
                  {isRecording && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
                      </span>
                      {t("recording.status.recording")}
                    </div>
                  )}

                  <div ref={transcriptEndRef} />
                </div>
              )}
            </div>
          </DDXCardContent>
        </DDXCard>
      </div>
    </div>
  );
}

function SegmentItem({ segment }: { segment: TranscriptionSegment }): React.ReactNode {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="group rounded-lg bg-muted/50 p-3 transition-colors hover:bg-muted">
      <div className="mb-1 flex items-center gap-2">
        {segment.speaker && (
          <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium text-white ${getSpeakerColor(segment.speaker)}`}>
            {segment.speaker}
          </span>
        )}
        <span className="font-mono text-xs text-muted-foreground">
          {formatTime(segment.start)} - {formatTime(segment.end)}
        </span>
        {segment.confidence !== undefined && (
          <span className="ml-auto text-xs text-muted-foreground">{Math.round(segment.confidence * 100)}%</span>
        )}
      </div>
      <p className="text-foreground">{segment.text}</p>
    </div>
  );
}
