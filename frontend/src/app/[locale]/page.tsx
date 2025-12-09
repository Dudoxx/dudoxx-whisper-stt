import { redirect } from "next/navigation";
import { setRequestLocale } from "next-intl/server";
import type { Locale } from "@/types";

interface HomePageProps {
  params: Promise<{ locale: Locale }>;
}

export default async function HomePage({
  params,
}: HomePageProps): Promise<never> {
  const { locale } = await params;
  setRequestLocale(locale);

  // Redirect to recordings page
  redirect(`/${locale}/recordings`);
}
