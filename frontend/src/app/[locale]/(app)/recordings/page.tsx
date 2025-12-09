"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useTranslations, useLocale } from "next-intl";
import { Mic, Play, Trash2, FileText, Clock, Plus } from "lucide-react";
import {
  DDXCard,
  DDXCardContent,
  DDXCardDescription,
  DDXCardHeader,
  DDXCardTitle,
  DDXButton,
  DDXBadge,
} from "@/components/ddx";
import { getAllRecordings, deleteRecording } from "@/lib/storage";
import type { Recording } from "@/types";

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("default", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(date));
}

export default function RecordingsPage(): React.ReactNode {
  const t = useTranslations();
  const locale = useLocale();
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecordings();
  }, []);

  async function loadRecordings(): Promise<void> {
    try {
      const data = await getAllRecordings();
      setRecordings(data);
    } catch (error) {
      console.error("Failed to load recordings:", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string): Promise<void> {
    if (!window.confirm(t("transcription.confirmDelete"))) {
      return;
    }

    try {
      await deleteRecording(id);
      setRecordings((prev) => prev.filter((r) => r.id !== id));
    } catch (error) {
      console.error("Failed to delete recording:", error);
    }
  }

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

  if (recordings.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
          <Mic className="h-8 w-8 text-muted-foreground" />
        </div>
        <h2 className="mt-4 text-xl font-semibold">
          {t("transcription.emptyState.title")}
        </h2>
        <p className="mt-2 text-center text-muted-foreground">
          {t("transcription.emptyState.description")}
        </p>
        <DDXButton asChild className="mt-6">
          <Link href={`/${locale}/record`}>
            <Plus className="mr-2 h-4 w-4" />
            {t("nav.newRecording")}
          </Link>
        </DDXButton>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("nav.recordings")}</h1>
          <p className="text-muted-foreground">
            {recordings.length} {t("transcription.words")}
          </p>
        </div>
        <DDXButton asChild>
          <Link href={`/${locale}/record`}>
            <Plus className="mr-2 h-4 w-4" />
            {t("nav.newRecording")}
          </Link>
        </DDXButton>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {recordings.map((recording) => (
          <DDXCard key={recording.id} className="group relative">
            <DDXCardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Mic className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <DDXCardTitle className="text-base">
                      {recording.name || `Recording ${recording.id.slice(0, 8)}`}
                    </DDXCardTitle>
                    <DDXCardDescription className="text-xs">
                      {formatDate(recording.createdAt)}
                    </DDXCardDescription>
                  </div>
                </div>
                {recording.language && (
                  <DDXBadge variant="secondary">{recording.language}</DDXBadge>
                )}
              </div>
            </DDXCardHeader>
            <DDXCardContent>
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatDuration(recording.duration)}
                </span>
                {recording.transcriptionId && (
                  <span className="flex items-center gap-1 text-success">
                    <FileText className="h-3 w-3" />
                    Transcribed
                  </span>
                )}
              </div>

              <div className="mt-4 flex gap-2">
                <DDXButton
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  asChild
                >
                  <Link href={`/${locale}/recordings/${recording.id}`}>
                    <Play className="mr-1 h-3 w-3" />
                    {t("playback.play")}
                  </Link>
                </DDXButton>
                <DDXButton
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(recording.id)}
                  className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </DDXButton>
              </div>
            </DDXCardContent>
          </DDXCard>
        ))}
      </div>
    </div>
  );
}
