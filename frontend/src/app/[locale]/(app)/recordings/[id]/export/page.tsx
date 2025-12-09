"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useTranslations, useLocale } from "next-intl";
import {
  ArrowLeft,
  Download,
  FileText,
  Code2,
  Subtitles,
  Check,
  Clock,
  User,
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
  getTranscriptionByRecording,
} from "@/lib/storage";
import {
  exportTranscription,
  downloadExport,
} from "@/lib/export";
import type { Recording, Transcription, ExportFormat, ExportOptions } from "@/types";

const FORMAT_OPTIONS: Array<{
  format: ExportFormat;
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
  description: string;
}> = [
  {
    format: "txt",
    icon: FileText,
    labelKey: "txt",
    description: "Simple text format, easy to read and share",
  },
  {
    format: "json",
    icon: Code2,
    labelKey: "json",
    description: "Structured data with all metadata and timestamps",
  },
  {
    format: "srt",
    icon: Subtitles,
    labelKey: "srt",
    description: "SubRip subtitle format for video players",
  },
  {
    format: "vtt",
    icon: Subtitles,
    labelKey: "vtt",
    description: "Web Video Text Tracks for HTML5 video",
  },
];

export default function ExportPage(): React.ReactNode {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const params = useParams();
  const recordingId = params.id as string;

  const [recording, setRecording] = useState<Recording | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>("txt");
  const [includeTimestamps, setIncludeTimestamps] = useState(true);
  const [includeSpeakers, setIncludeSpeakers] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [preview, setPreview] = useState<string>("");

  // Load recording and transcription
  useEffect(() => {
    async function loadData(): Promise<void> {
      try {
        setLoading(true);
        setError(null);

        const [rec, trans] = await Promise.all([
          getRecording(recordingId),
          getTranscriptionByRecording(recordingId),
        ]);

        if (!rec) {
          setError("Recording not found");
          return;
        }

        if (!trans) {
          setError("No transcription available for this recording");
          return;
        }

        setRecording(rec);
        setTranscription(trans);
      } catch (err) {
        console.error("Failed to load recording:", err);
        setError("Failed to load recording");
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [recordingId]);

  // Update preview when options change
  useEffect(() => {
    if (!transcription) return;

    const options: ExportOptions = {
      format: selectedFormat,
      includeTimestamps,
      includeSpeakers,
    };

    try {
      const content = exportTranscription(transcription, options);
      // Limit preview to first 1000 characters
      setPreview(content.length > 1000 ? `${content.slice(0, 1000)}...` : content);
    } catch (err) {
      console.error("Failed to generate preview:", err);
      setPreview("");
    }
  }, [transcription, selectedFormat, includeTimestamps, includeSpeakers]);

  const handleExport = useCallback(async () => {
    if (!transcription || !recording) return;

    setIsExporting(true);
    setExportSuccess(false);

    try {
      const options: ExportOptions = {
        format: selectedFormat,
        includeTimestamps,
        includeSpeakers,
      };

      const content = exportTranscription(transcription, options);
      const filename = recording.name || `recording_${recording.id.slice(0, 8)}`;

      downloadExport(content, filename, selectedFormat);
      setExportSuccess(true);

      // Reset success state after 3 seconds
      setTimeout(() => setExportSuccess(false), 3000);
    } catch (err) {
      console.error("Failed to export:", err);
      setError("Failed to export transcription");
    } finally {
      setIsExporting(false);
    }
  }, [transcription, recording, selectedFormat, includeTimestamps, includeSpeakers]);

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

  if (error || !recording || !transcription) {
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <DDXButton
          variant="ghost"
          size="icon"
          onClick={() => router.push(`/${locale}/recordings/${recordingId}`)}
        >
          <ArrowLeft className="h-5 w-5" />
        </DDXButton>
        <div>
          <h1 className="text-2xl font-bold">{t("export.title")}</h1>
          <p className="text-sm text-muted-foreground">
            {recording.name || `Recording ${recording.id.slice(0, 8)}`}
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Options Column */}
        <div className="space-y-6">
          {/* Format Selection */}
          <DDXCard>
            <DDXCardHeader>
              <DDXCardTitle>{t("export.format")}</DDXCardTitle>
            </DDXCardHeader>
            <DDXCardContent className="space-y-3">
              {FORMAT_OPTIONS.map((option) => {
                const Icon = option.icon;
                const isSelected = selectedFormat === option.format;

                return (
                  <button
                    key={option.format}
                    onClick={() => setSelectedFormat(option.format)}
                    className={`flex w-full items-start gap-4 rounded-lg border p-4 text-left transition-all ${
                      isSelected
                        ? "border-primary bg-primary/5 ring-2 ring-primary"
                        : "border-border hover:border-primary/50 hover:bg-muted/50"
                    }`}
                  >
                    <div
                      className={`flex h-10 w-10 items-center justify-center rounded-lg ${
                        isSelected ? "bg-primary text-primary-foreground" : "bg-muted"
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">
                          {t(`export.formats.${option.labelKey}`)}
                        </span>
                        {isSelected && (
                          <Check className="h-4 w-4 text-primary" />
                        )}
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {option.description}
                      </p>
                    </div>
                  </button>
                );
              })}
            </DDXCardContent>
          </DDXCard>

          {/* Export Options */}
          <DDXCard>
            <DDXCardHeader>
              <DDXCardTitle>Options</DDXCardTitle>
            </DDXCardHeader>
            <DDXCardContent className="space-y-4">
              {/* Include Timestamps */}
              <label className="flex cursor-pointer items-center justify-between rounded-lg border border-border p-4 hover:bg-muted/50">
                <div className="flex items-center gap-3">
                  <Clock className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{t("export.includeTimestamps")}</p>
                    <p className="text-sm text-muted-foreground">
                      Add timing information to each segment
                    </p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={includeTimestamps}
                  onChange={(e) => setIncludeTimestamps(e.target.checked)}
                  className="h-5 w-5 rounded border-input accent-primary"
                />
              </label>

              {/* Include Speakers */}
              <label className="flex cursor-pointer items-center justify-between rounded-lg border border-border p-4 hover:bg-muted/50">
                <div className="flex items-center gap-3">
                  <User className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">{t("export.includeSpeakers")}</p>
                    <p className="text-sm text-muted-foreground">
                      Add speaker labels to distinguish voices
                    </p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={includeSpeakers}
                  onChange={(e) => setIncludeSpeakers(e.target.checked)}
                  className="h-5 w-5 rounded border-input accent-primary"
                />
              </label>
            </DDXCardContent>
          </DDXCard>

          {/* Export Button */}
          <DDXButton
            onClick={handleExport}
            loading={isExporting}
            disabled={!transcription}
            size="lg"
            className="w-full"
            variant={exportSuccess ? "success" : "default"}
          >
            {exportSuccess ? (
              <>
                <Check className="mr-2 h-5 w-5" />
                Downloaded!
              </>
            ) : (
              <>
                <Download className="mr-2 h-5 w-5" />
                {t("export.download")}
              </>
            )}
          </DDXButton>
        </div>

        {/* Preview Column */}
        <DDXCard className="flex flex-col">
          <DDXCardHeader className="border-b">
            <DDXCardTitle className="flex items-center justify-between">
              <span>Preview</span>
              <DDXBadge variant="secondary">
                {t(`export.formats.${selectedFormat}`)}
              </DDXBadge>
            </DDXCardTitle>
          </DDXCardHeader>
          <DDXCardContent className="flex-1 p-0">
            <pre className="h-full min-h-[400px] overflow-auto bg-muted/30 p-4 font-mono text-sm">
              {preview || "No preview available"}
            </pre>
          </DDXCardContent>
        </DDXCard>
      </div>
    </div>
  );
}
