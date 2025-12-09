"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { useLocale } from "next-intl";
import { useRouter, usePathname } from "next/navigation";
import { Moon, Sun, ChevronDown, Check } from "lucide-react";
import { DDXButton } from "@/components/ddx";
import {
  DDXDropdownMenu,
  DDXDropdownMenuTrigger,
  DDXDropdownMenuContent,
  DDXDropdownMenuItem,
} from "@/components/ddx";
import type { Locale } from "@/types";

const locales: { code: Locale; name: string; flag: string }[] = [
  { code: "en", name: "English", flag: "🇬🇧" },
  { code: "fr", name: "Français", flag: "🇫🇷" },
  { code: "de", name: "Deutsch", flag: "🇩🇪" },
];

export function AppHeader(): React.ReactNode {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  
  // Prevent hydration mismatch
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleLocaleChange = (newLocale: Locale): void => {
    const newPathname = pathname.replace(`/${locale}`, `/${newLocale}`);
    router.push(newPathname);
  };

  const toggleTheme = (): void => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  };

  const currentLocale = locales.find((l) => l.code === locale) || locales[0];

  return (
    <header className="flex h-14 items-center justify-end border-b border-border bg-card px-4">
      <div className="flex items-center gap-2">
        {/* Locale Dropdown */}
        <DDXDropdownMenu>
          <DDXDropdownMenuTrigger asChild>
            <DDXButton variant="ghost" size="sm" className="gap-1.5">
              <span className="text-base">{currentLocale.flag}</span>
              <span className="hidden sm:inline">{currentLocale.code.toUpperCase()}</span>
              <ChevronDown className="h-3.5 w-3.5 opacity-50" />
            </DDXButton>
          </DDXDropdownMenuTrigger>
          <DDXDropdownMenuContent align="end">
            {locales.map((loc) => (
              <DDXDropdownMenuItem
                key={loc.code}
                onClick={() => handleLocaleChange(loc.code)}
                className="gap-2"
              >
                <span className="text-base">{loc.flag}</span>
                <span>{loc.name}</span>
                {locale === loc.code && (
                  <Check className="ml-auto h-4 w-4" />
                )}
              </DDXDropdownMenuItem>
            ))}
          </DDXDropdownMenuContent>
        </DDXDropdownMenu>

        {/* Theme Toggle - Sun/Moon */}
        <DDXButton
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {mounted ? (
            resolvedTheme === "dark" ? (
              <Moon className="h-4 w-4" />
            ) : (
              <Sun className="h-4 w-4" />
            )
          ) : (
            <Sun className="h-4 w-4" />
          )}
        </DDXButton>
      </div>
    </header>
  );
}
