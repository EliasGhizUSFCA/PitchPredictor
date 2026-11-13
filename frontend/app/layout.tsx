import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/context";
import { Shell } from "@/components/shell";

export const metadata: Metadata = {
  title: "PitchIQ AI",
  description: "MLB next-pitch prediction research platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppProvider>
          <Shell>{children}</Shell>
        </AppProvider>
      </body>
    </html>
  );
}
