"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useTranslations, useLocale } from "next-intl";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Download,
  ArrowLeft,
  Clock,
  FileText,
  Trash2,
  Copy,
  Check,
} from "lucide-react";
import {
  DDXButton,
  DDXCard,
  DDXCardContent,
  DDXCardHeader,
  DDXCardTitle,
  DDXBadge,
} from "@/components/ddx";
import {
  getRecording,
  getAudioBlob,
  getTranscription,
  deleteRecording,
  deleteTranscription,
} from "@/lib/storage";
import type { Recording, Transcription, TranscriptionSegment } from "@/types";

// Speaker colors
const SPEAKER_COLORS = [
  "bg-speaker-1",
  "bg-speaker-2",
  "bg-speaker-3",
  "bg-speaker-4",
];

function getSpeakerColor(speaker: string | undefined): string {
  if (!speaker) return "bg-muted";
  const speakerNum = parseInt(speaker.replace(/\D/g, ""), 10) || 0;
  return SPEAKER_COLORS[speakerNum % SPEAKER_COLORS.length];
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("default", {
    dateStyle: "long",
    timeStyle: "short",
  }).format(new Date(date));
}

export default function PlaybackPage(): React.ReactNode {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const params = useParams();
  const recordingId = params.id as string;

  const audioRef = useRef<HTMLAudioElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  const [recording, setRecording] = useState<Recording | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Load recording and transcription
  useEffect(() => {
    async function loadData(): Promise<void> {
      try {
        setLoading(true);
        setError(null);

        // First get recording
        const rec = await getRecording(recordingId);
        if (!rec) {
          setError("Recording not found");
          return;
        }
        setRecording(rec);

        // Get transcription using the transcriptionId from recording
        if (rec.transcriptionId) {
          const trans = await getTranscription(rec.transcriptionId);
          console.log("[PlaybackPage] Loaded transcription:", trans?.id, "segments:", trans?.segments?.length);
          setTranscription(trans || null);
        }

        // Get audio blob
        const blob = await getAudioBlob(recordingId);
        if (blob) {
          const url = URL.createObjectURL(blob);
          setAudioUrl(url);
        }
      } catch (err) {
        console.error("Failed to load recording:", err);
        setError("Failed to load recording");
      } finally {
        setLoading(false);
      }
    }

    loadData();

    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [recordingId]);

  // Update active segment based on current time
  useEffect(() => {
    if (!transcription) return;

    const segment = transcription.segments.find(
      (s) => currentTime >= s.start && currentTime <= s.end
    );

    if (segment && segment.id !== activeSegmentId) {
      setActiveSegmentId(segment.id);
    }
  }, [currentTime, transcription, activeSegmentId]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (!activeSegmentId || !transcriptRef.current) return;

    const activeElement = transcriptRef.current.querySelector(
      `[data-segment-id="${activeSegmentId}"]`
    );

    if (activeElement) {
      activeElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [activeSegmentId]);

  // Audio event handlers
  const handleTimeUpdate = useCallback(() => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  }, []);

  const handleLoadedMetadata = useCallback(() => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  }, []);

  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, []);

  // Playback controls
  const togglePlay = useCallback(() => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const seekTo = useCallback((time: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(time, duration));
    setCurrentTime(audioRef.current.currentTime);
  }, [duration]);

  const skipBackward = useCallback(() => {
    seekTo(currentTime - 10);
  }, [currentTime, seekTo]);

  const skipForward = useCallback(() => {
    seekTo(currentTime + 10);
  }, [currentTime, seekTo]);

  const toggleMute = useCallback(() => {
    if (!audioRef.current) return;
    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  }, []);

  const handlePlaybackRateChange = useCallback((rate: number) => {
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
    setPlaybackRate(rate);
  }, []);

  const handleSeekChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    seekTo(newTime);
  }, [seekTo]);

  const handleSegmentClick = useCallback((segment: TranscriptionSegment) => {
    seekTo(segment.start);
    if (!isPlaying && audioRef.current) {
      audioRef.current.play();
      setIsPlaying(true);
    }
  }, [seekTo, isPlaying]);

  const handleCopyTranscript = useCallback(async () => {
    if (!transcription) return;
    
    try {
      await navigator.clipboard.writeText(transcription.fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  }, [transcription]);

  const handleDelete = useCallback(async () => {
    if (!window.confirm(t("transcription.confirmDelete"))) return;

    try {
      await deleteRecording(recordingId);
      if (transcription) {
        await deleteTranscription(transcription.id);
      }
      router.push(`/${locale}/recordings`);
    } catch (err) {
      console.error("Failed to delete:", err);
    }
  }, [recordingId, transcription, locale, router, t]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="mt-2 text-sm text-muted-foreground">{t("common.loading")}</p>
        </div>
      </div>
    );
  }

  if (error || !recording) {
    return (
      <div className="flex h-full flex-col items-center justify-center">
        <p className="text-destructive">{error || "Recording not found"}</p>
        <DDXButton
          variant="outline"
          className="mt-4"
          onClick={() => router.push(`/${locale}/recordings`)}
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          {t("common.back")}
        </DDXButton>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-6">
      {/* Hidden audio element */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          preload="metadata"
        />
      )}

      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <DDXButton
            variant="ghost"
            size="icon"
            onClick={() => router.push(`/${locale}/recordings`)}
          >
            <ArrowLeft className="h-5 w-5" />
          </DDXButton>
          <div>
            <h1 className="text-2xl font-bold">
              {recording.name || `Recording ${recording.id.slice(0, 8)}`}
            </h1>
            <p className="text-sm text-muted-foreground">
              {formatDate(recording.createdAt)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {recording.language && (
            <DDXBadge variant="secondary">{recording.language}</DDXBadge>
          )}
          <DDXButton variant="ghost" size="icon" onClick={handleDelete}>
            <Trash2 className="h-4 w-4 text-destructive" />
          </DDXButton>
        </div>
      </div>

      {/* Main content grid */}
      <div className="grid flex-1 gap-6 lg:grid-cols-3">
        {/* Player Column */}
        <DDXCard className="lg:col-span-1">
          <DDXCardHeader>
            <DDXCardTitle>{t("playback.title")}</DDXCardTitle>
          </DDXCardHeader>
          <DDXCardContent className="space-y-6">
            {/* Time Display */}
            <div className="text-center">
              <div className="font-mono text-4xl font-bold tabular-nums">
                {formatTime(currentTime)}
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                / {formatTime(duration || recording.duration)}
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <input
                type="range"
                min={0}
                max={duration || recording.duration}
                step={0.1}
                value={currentTime}
                onChange={handleSeekChange}
                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              />
            </div>

            {/* Playback Controls */}
            <div className="flex items-center justify-center gap-4">
              <DDXButton
                variant="ghost"
                size="icon"
                onClick={skipBackward}
                disabled={!audioUrl}
              >
                <SkipBack className="h-5 w-5" />
              </DDXButton>
              <DDXButton
                variant="default"
                size="icon-lg"
                onClick={togglePlay}
                disabled={!audioUrl}
                className="h-14 w-14 rounded-full"
              >
                {isPlaying ? (
                  <Pause className="h-6 w-6" />
                ) : (
                  <Play className="h-6 w-6 ml-0.5" />
                )}
              </DDXButton>
              <DDXButton
                variant="ghost"
                size="icon"
                onClick={skipForward}
                disabled={!audioUrl}
              >
                <SkipForward className="h-5 w-5" />
              </DDXButton>
            </div>

            {/* Volume Control */}
            <div className="flex items-center gap-3">
              <DDXButton
                variant="ghost"
                size="icon-sm"
                onClick={toggleMute}
              >
                {isMuted || volume === 0 ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume2 className="h-4 w-4" />
                )}
              </DDXButton>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              />
            </div>

            {/* Playback Speed */}
            <div className="space-y-2">
              <label className="text-sm font-medium">{t("playback.speed")}</label>
              <div className="flex gap-1">
                {[0.5, 0.75, 1, 1.25, 1.5, 2].map((rate) => (
                  <DDXButton
                    key={rate}
                    variant={playbackRate === rate ? "default" : "outline"}
                    size="sm"
                    onClick={() => handlePlaybackRateChange(rate)}
                    className="flex-1 text-xs"
                  >
                    {rate}x
                  </DDXButton>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div className="space-y-2 rounded-lg bg-muted/50 p-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1 text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {t("recording.duration")}
                </span>
                <span className="font-mono">{formatTime(recording.duration)}</span>
              </div>
              {transcription && (
                <>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1 text-muted-foreground">
                      <FileText className="h-3 w-3" />
                      {t("transcription.words")}
                    </span>
                    <span>{transcription.metadata?.wordCount || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">
                      {t("transcription.sentences")}
                    </span>
                    <span>{transcription.segments.length}</span>
                  </div>
                </>
              )}
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <DDXButton
                variant="outline"
                className="flex-1"
                onClick={handleCopyTranscript}
                disabled={!transcription}
              >
                {copied ? (
                  <>
                    <Check className="mr-2 h-4 w-4" />
                    {t("transcription.copied")}
                  </>
                ) : (
                  <>
                    <Copy className="mr-2 h-4 w-4" />
                    {t("transcription.copy")}
                  </>
                )}
              </DDXButton>
              <DDXButton
                variant="outline"
                onClick={() => router.push(`/${locale}/recordings/${recordingId}/export`)}
                disabled={!transcription}
              >
                <Download className="h-4 w-4" />
              </DDXButton>
            </div>
          </DDXCardContent>
        </DDXCard>

        {/* Transcription Column */}
        <DDXCard className="flex flex-col lg:col-span-2">
          <DDXCardHeader className="border-b">
            <DDXCardTitle className="flex items-center justify-between">
              <span>{t("transcription.title")}</span>
              {transcription && (
                <DDXBadge variant="secondary">
                  {transcription.segments.length} {t("transcription.sentences")}
                </DDXBadge>
              )}
            </DDXCardTitle>
          </DDXCardHeader>
          <DDXCardContent className="flex-1 overflow-auto p-0">
            <div ref={transcriptRef} className="h-full min-h-[400px] p-4">
              {!transcription || transcription.segments.length === 0 ? (
                <div className="flex h-full flex-col items-center justify-center text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                    <FileText className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="mt-4 text-muted-foreground">
                    {t("transcription.noTranscription")}
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {transcription.segments.map((segment) => (
                    <button
                      key={segment.id}
                      data-segment-id={segment.id}
                      onClick={() => handleSegmentClick(segment)}
                      className={`w-full rounded-lg p-3 text-left transition-all ${
                        activeSegmentId === segment.id
                          ? "bg-primary/10 ring-2 ring-primary"
                          : "bg-muted/50 hover:bg-muted"
                      }`}
                    >
                      <div className="mb-1 flex items-center gap-2">
                        {segment.speaker && (
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium text-white ${getSpeakerColor(segment.speaker)}`}
                          >
                            {segment.speaker}
                          </span>
                        )}
                        <span className="font-mono text-xs text-muted-foreground">
                          {formatTime(segment.start)} - {formatTime(segment.end)}
                        </span>
                        {segment.confidence !== undefined && (
                          <span className="ml-auto text-xs text-muted-foreground">
                            {Math.round(segment.confidence * 100)}%
                          </span>
                        )}
                      </div>
                      <p
                        className={`text-foreground ${
                          activeSegmentId === segment.id ? "font-medium" : ""
                        }`}
                      >
                        {segment.text}
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </DDXCardContent>
        </DDXCard>
      </div>
    </div>
  );
}
