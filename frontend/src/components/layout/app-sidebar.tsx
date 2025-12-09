"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTranslations } from "next-intl";
import {
  Mic,
  FileText,
  Settings,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils/cn";
import { DDXButton } from "@/components/ddx";

interface NavItem {
  href: string;
  labelKey: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { href: "/recordings", labelKey: "nav.recordings", icon: Mic },
  { href: "/transcriptions", labelKey: "nav.transcriptions", icon: FileText },
  { href: "/settings", labelKey: "nav.settings", icon: Settings },
];

export function AppSidebar(): React.ReactNode {
  const t = useTranslations();
  const pathname = usePathname();
  
  // Extract locale from pathname (e.g., /en/recordings -> en)
  const localeMatch = pathname.match(/^\/([a-z]{2})\//);
  const locale = localeMatch ? localeMatch[1] : "en";

  return (
    <aside className="flex h-full w-64 flex-col border-r border-border bg-card">
      {/* Header */}
      <div className="flex h-16 items-center border-b border-border px-4">
        <Link href={`/${locale}`} className="flex items-center gap-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/logo.png"
            alt="Dudoxx Transcriber"
            width={32}
            height={32}
            className="h-8 w-8"
          />
          <span className="font-semibold">{t("app.name")}</span>
        </Link>
      </div>

      {/* New Recording Button */}
      <div className="p-4">
        <DDXButton asChild className="w-full" size="lg">
          <Link href={`/${locale}/record`}>
            <Plus className="h-4 w-4" />
            {t("nav.newRecording")}
          </Link>
        </DDXButton>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3">
        {navItems.map((item) => {
          const href = `/${locale}${item.href}`;
          const isActive = pathname.startsWith(href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {t(item.labelKey)}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-border p-4">
        <p className="text-xs text-muted-foreground">
          {t("app.name")} v1.0.0
        </p>
      </div>
    </aside>
  );
}
