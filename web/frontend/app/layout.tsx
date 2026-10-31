import type { Metadata, Viewport } from "next";
import { Amiri, IBM_Plex_Sans_Arabic } from "next/font/google";
import "@openuidev/react-ui/layered/styles/index.css";
import "./globals.css";

const sans = IBM_Plex_Sans_Arabic({
  subsets: ["arabic", "latin"],
  weight: ["300", "400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

// Classical text: the passages and quotations.
const naskh = Amiri({
  subsets: ["arabic", "latin"],
  weight: ["400", "700"],
  variable: "--font-naskh",
  display: "swap",
});

export const metadata: Metadata = {
  title: "دليل — بحث موثّق في كتب التراث",
  description: "Arabic-first retrieval over classical Islamic texts, with cited answers.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  // Android Chrome: the on-screen keyboard shrinks the page, so the question box stays above it.
  interactiveWidget: "resizes-content",
  themeColor: "#f7f5f0",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ar" dir="rtl" className={`${sans.variable} ${naskh.variable}`}>
      <head>
        {/* @openuidev/react-lang auto-mounts its Inspect widget in development, fetched from
            jsDelivr. Setting its flag first keeps it off; NEXT_PUBLIC_OPENUI_DEVTOOLS=1 brings it back. */}
        {process.env.NEXT_PUBLIC_OPENUI_DEVTOOLS === "1" ? null : (
          <script
            dangerouslySetInnerHTML={{ __html: 'globalThis[Symbol.for("openui.devtools.autoMount")]=true;' }}
          />
        )}
      </head>
      <body>{children}</body>
    </html>
  );
}
