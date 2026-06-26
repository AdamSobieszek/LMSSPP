import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LMS Exact-Gauge Framework Map",
  description:
    "Interactive dependency graph connecting LMS hyperbolic reduction, exact Busemann inversion, canonical gauges, invariants, and continuum perspectives.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
