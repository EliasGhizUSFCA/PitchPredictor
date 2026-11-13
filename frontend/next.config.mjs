/** @type {import('next').NextConfig} */
const API = process.env.PITCHIQ_API_URL || "http://localhost:8000";
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};
export default nextConfig;
