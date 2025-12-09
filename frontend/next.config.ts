import createNextIntlPlugin from 'next-intl/plugin';
import type { NextConfig } from 'next';

const withNextIntl = createNextIntlPlugin('./src/lib/i18n/request.ts');

const nextConfig: NextConfig = {
  // Build output directory - .next for dev, .next_prod for production
  distDir: process.env.NODE_ENV === 'production' ? '.next_prod' : '.next',

  reactStrictMode: true,

  typescript: {
    ignoreBuildErrors: false,
  },

  // ESLint config is now in eslint.config.mjs for Next.js 16

  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },

  // Environment variables exposed to the client
  env: {
    NEXT_PUBLIC_WHISPER_STT_URL: process.env.NEXT_PUBLIC_WHISPER_STT_URL || 'http://localhost:4300',
  },
};

export default withNextIntl(nextConfig);
