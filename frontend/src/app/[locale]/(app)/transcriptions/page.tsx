"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useTranslations, useLocale } from "next-intl";
import { FileText, Download, Trash2, Clock, Users } from "lucide-react";
import {
  DDXCard,
  DDXCardContent,
  DDXCardDescription,
  DDXCardHeader,
  DDXCardTitle,
  DDXButton,
  DDXBadge,
} from "@/components/ddx";
import { getAllTranscriptions, deleteTranscription } from "@/lib/storage";
import type { Transcription } from "@/types";

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("default", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(date));
}

export default function TranscriptionsPage(): React.ReactNode {
  const t = useTranslations();
  const locale = useLocale();
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTranscriptions();
  }, []);

  async function loadTranscriptions(): Promise<void> {
    try {
      const data = await getAllTranscriptions();
      setTranscriptions(data);
    } catch (error) {
      console.error("Failed to load transcriptions:", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: string): Promise<void> {
    if (!window.confirm(t("transcription.confirmDelete"))) {
      return;
    }

    try {
      await deleteTranscription(id);
      setTranscriptions((prev) => prev.filter((t) => t.id !== id));
    } catch (error) {
      console.error("Failed to delete transcription:", error);
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

  if (transcriptions.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
          <FileText className="h-8 w-8 text-muted-foreground" />
        </div>
        <h2 className="mt-4 text-xl font-semibold">
          {t("transcription.emptyState.title")}
        </h2>
        <p className="mt-2 text-center text-muted-foreground">
          {t("transcription.emptyState.description")}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("nav.transcriptions")}</h1>
          <p className="text-muted-foreground">
            {transcriptions.length} {t("transcription.words")}
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {transcriptions.map((transcription) => (
          <DDXCard key={transcription.id} className="group relative">
            <DDXCardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <DDXCardTitle className="text-base">
                      Transcription {transcription.id.slice(0, 8)}
                    </DDXCardTitle>
                    <DDXCardDescription className="text-xs">
                      {formatDate(transcription.createdAt)}
                    </DDXCardDescription>
                  </div>
                </div>
                <DDXBadge variant="secondary">{transcription.language}</DDXBadge>
              </div>
            </DDXCardHeader>
            <DDXCardContent>
              <p className="line-clamp-3 text-sm text-muted-foreground">
                {transcription.fullText.slice(0, 150)}
                {transcription.fullText.length > 150 ? "..." : ""}
              </p>

              <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {transcription.metadata?.wordCount || 0} {t("transcription.words")}
                </span>
                <span className="flex items-center gap-1">
                  <Users className="h-3 w-3" />
                  {transcription.speakers.length} speakers
                </span>
              </div>

              <div className="mt-4 flex gap-2">
                <DDXButton
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  asChild
                >
                  <Link href={`/${locale}/transcriptions/${transcription.id}`}>
                    View
                  </Link>
                </DDXButton>
                <DDXButton variant="ghost" size="sm">
                  <Download className="h-4 w-4" />
                </DDXButton>
                <DDXButton
                  variant="ghost"
                  size="sm"
                  onClick={() => handleDelete(transcription.id)}
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
