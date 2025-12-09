import createMiddleware from "next-intl/middleware";
import { type NextRequest } from "next/server";
import { routing } from "./lib/i18n/routing";

const handleI18nRouting = createMiddleware(routing);

export function proxy(request: NextRequest) {
  return handleI18nRouting(request);
}

export const config = {
  // Match all pathnames except for
  // - api routes
  // - _next (Next.js internals)
  // - _vercel (Vercel internals)
  // - static files (images, etc.)
  matcher: "/((?!api|trpc|_next|_vercel|.*\\..*)*)",
};
