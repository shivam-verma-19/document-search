/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // 🔥 Backend API URL (Lambda / API Gateway)
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;