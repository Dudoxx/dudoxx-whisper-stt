import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Dudoxx Transcriber",
  description: "Real-time speech-to-text transcription with speaker diarization",
  keywords: ["speech-to-text", "transcription", "diarization", "whisper", "audio"],
  authors: [{ name: "Dudoxx" }],
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>): React.ReactNode {
  return children;
}
