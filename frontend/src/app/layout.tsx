import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VocalSense AI — Speech Emotion Recognition",
  description:
    "Advanced deep learning speech signal processing and emotion classification from acoustic MFCCs using CNN, Bi-LSTM, and CRNN architectures.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen bg-[#090e1a] text-slate-100">
        {children}
      </body>
    </html>
  );
}
