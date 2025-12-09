"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { useTheme } from "next-themes";
import { Sun, Moon, Monitor, Server, Database, Trash2 } from "lucide-react";
import {
  DDXCard,
  DDXCardContent,
  DDXCardDescription,
  DDXCardHeader,
  DDXCardTitle,
  DDXButton,
  DDXInput,
  DDXBadge,
} from "@/components/ddx";
import {
  DDXDialog,
  DDXDialogContent,
  DDXDialogDescription,
  DDXDialogFooter,
  DDXDialogHeader,
  DDXDialogTitle,
  DDXDialogTrigger,
} from "@/components/ddx";
import { getSettings, saveSettings, getStorageInfo, clearAllData } from "@/lib/storage";
import type { AppSettings } from "@/types";

const DEFAULT_SETTINGS: AppSettings = {
  serverUrl: "ws://localhost:4300/ws",
  language: "auto",
  theme: "system",
  autoSave: true,
  showTimestamps: true,
  showSpeakers: true,
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export default function SettingsPage(): React.ReactNode {
  const t = useTranslations("settings");
  const tCommon = useTranslations("common");
  const { theme, setTheme } = useTheme();
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [storageInfo, setStorageInfo] = useState({
    used: 0,
    recordings: 0,
    transcriptions: 0,
  });
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "disconnected" | "testing">(
    "disconnected",
  );
  const [loading, setLoading] = useState(true);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  useEffect(() => {
    loadSettings();
    loadStorageInfo();
  }, []);

  async function loadSettings(): Promise<void> {
    try {
      const saved = await getSettings();
      if (saved) {
        setSettings(saved);
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    } finally {
      setLoading(false);
    }
  }

  async function loadStorageInfo(): Promise<void> {
    try {
      const info = await getStorageInfo();
      setStorageInfo(info);
    } catch (error) {
      console.error("Failed to load storage info:", error);
    }
  }

  async function handleSave(): Promise<void> {
    try {
      await saveSettings(settings);
    } catch (error) {
      console.error("Failed to save settings:", error);
    }
  }

  async function handleTestConnection(): Promise<void> {
    setConnectionStatus("testing");
    try {
      const ws = new WebSocket(settings.serverUrl);
      ws.onopen = () => {
        setConnectionStatus("connected");
        ws.close();
      };
      ws.onerror = () => {
        setConnectionStatus("disconnected");
      };
      // Timeout after 5 seconds
      setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          setConnectionStatus("disconnected");
          ws.close();
        }
      }, 5000);
    } catch {
      setConnectionStatus("disconnected");
    }
  }

  async function handleClearData(): Promise<void> {
    try {
      await clearAllData();
      await loadStorageInfo();
      setClearDialogOpen(false);
    } catch (error) {
      console.error("Failed to clear data:", error);
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t("title")}</h1>
        <p className="text-muted-foreground">{t("general")}</p>
      </div>

      {/* Appearance */}
      <DDXCard>
        <DDXCardHeader>
          <DDXCardTitle>{t("appearance")}</DDXCardTitle>
          <DDXCardDescription>{t("theme")}</DDXCardDescription>
        </DDXCardHeader>
        <DDXCardContent>
          <div className="flex gap-2">
            <DDXButton
              variant={theme === "light" ? "default" : "outline"}
              className="flex-1 gap-2"
              onClick={() => setTheme("light")}
            >
              <Sun className="h-4 w-4" />
              {t("themes.light")}
            </DDXButton>
            <DDXButton
              variant={theme === "dark" ? "default" : "outline"}
              className="flex-1 gap-2"
              onClick={() => setTheme("dark")}
            >
              <Moon className="h-4 w-4" />
              {t("themes.dark")}
            </DDXButton>
            <DDXButton
              variant={theme === "system" ? "default" : "outline"}
              className="flex-1 gap-2"
              onClick={() => setTheme("system")}
            >
              <Monitor className="h-4 w-4" />
              {t("themes.system")}
            </DDXButton>
          </div>
        </DDXCardContent>
      </DDXCard>

      {/* Server Configuration */}
      <DDXCard>
        <DDXCardHeader>
          <div className="flex items-center justify-between">
            <div>
              <DDXCardTitle className="flex items-center gap-2">
                <Server className="h-4 w-4" />
                {t("server.title")}
              </DDXCardTitle>
              <DDXCardDescription>{t("server.url")}</DDXCardDescription>
            </div>
            <DDXBadge
              variant={connectionStatus === "connected" ? "connected" : "disconnected"}
            >
              {connectionStatus === "testing"
                ? tCommon("loading")
                : connectionStatus === "connected"
                  ? t("server.connected")
                  : t("server.disconnected")}
            </DDXBadge>
          </div>
        </DDXCardHeader>
        <DDXCardContent className="space-y-4">
          <DDXInput
            value={settings.serverUrl}
            onChange={(e) =>
              setSettings((prev) => ({ ...prev, serverUrl: e.target.value }))
            }
            placeholder="ws://localhost:4300/ws"
          />
          <div className="flex gap-2">
            <DDXButton variant="outline" onClick={handleTestConnection}>
              {t("server.testConnection")}
            </DDXButton>
            <DDXButton onClick={handleSave}>{tCommon("save")}</DDXButton>
          </div>
        </DDXCardContent>
      </DDXCard>

      {/* Storage */}
      <DDXCard>
        <DDXCardHeader>
          <DDXCardTitle className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            {t("storage.title")}
          </DDXCardTitle>
          <DDXCardDescription>
            {t("storage.used")}: {formatBytes(storageInfo.used)}
          </DDXCardDescription>
        </DDXCardHeader>
        <DDXCardContent>
          <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
            <div className="rounded-lg bg-muted p-3">
              <p className="text-muted-foreground">Recordings</p>
              <p className="text-xl font-semibold">{storageInfo.recordings}</p>
            </div>
            <div className="rounded-lg bg-muted p-3">
              <p className="text-muted-foreground">Transcriptions</p>
              <p className="text-xl font-semibold">{storageInfo.transcriptions}</p>
            </div>
          </div>

          <DDXDialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
            <DDXDialogTrigger asChild>
              <DDXButton variant="destructive" className="w-full gap-2">
                <Trash2 className="h-4 w-4" />
                {t("storage.clearAll")}
              </DDXButton>
            </DDXDialogTrigger>
            <DDXDialogContent>
              <DDXDialogHeader>
                <DDXDialogTitle>{t("storage.clearAll")}</DDXDialogTitle>
                <DDXDialogDescription>{t("storage.confirmClear")}</DDXDialogDescription>
              </DDXDialogHeader>
              <DDXDialogFooter>
                <DDXButton
                  variant="outline"
                  onClick={() => setClearDialogOpen(false)}
                >
                  {tCommon("cancel")}
                </DDXButton>
                <DDXButton variant="destructive" onClick={handleClearData}>
                  {tCommon("delete")}
                </DDXButton>
              </DDXDialogFooter>
            </DDXDialogContent>
          </DDXDialog>
        </DDXCardContent>
      </DDXCard>
    </div>
  );
}
