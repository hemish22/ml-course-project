import type { Metadata } from "next";
import { Bricolage_Grotesque, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  variable: "--font-sans",
  subsets: ["latin"],
});

const timecodeFont = JetBrains_Mono({
  variable: "--font-time",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Needle – search inside video",
  description:
    "Find moments in a video by describing what you see or what is said. Picture and speech are scored separately, then fused.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${bricolage.variable} ${timecodeFont.variable}`}>
      <body>{children}</body>
    </html>
  );
}
