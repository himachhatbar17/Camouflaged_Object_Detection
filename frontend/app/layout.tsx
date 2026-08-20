import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "\u0394Fusion \u2014 CamoNet Camouflaged Object Detection",
  description:
    "\u0394T-Guided Cross-Spectral Transfer for camouflaged object detection. Upload an image, run CamoNet, and inspect the predicted mask, confidence, and attention heatmap.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
