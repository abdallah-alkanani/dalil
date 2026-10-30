export type Category = "quran" | "hadith" | "tafsir" | "fiqh" | "fiqh_usul" | "aqeedah" | "seerah";

/** One search result as the dalil API returns it (`compact_result`), numbered in rank order. */
export interface Passage {
  n: number;
  parent_id: string;
  citation: string | null;
  category: Category | null;
  book_id: string | null;
  section_path: string[];
  page_number: number | null;
  refs: {
    hadith?: { collection?: string; hadith_number?: string } | null;
    quran?: { surah?: number; ayah?: number } | null;
  } & Record<string, unknown>;
  grade: string | null;
  text: string;
  truncated: boolean;
  retrieval_sources: string[];
  neighbours: { parent_id: string; offset: number; citation: string | null; text: string }[];
}

export interface Retrieval {
  query_id: number;
  status: Record<string, string>;
  latency_ms: number;
  passages: Passage[];
}

/** The NDJSON events /api/ask streams. */
export type AskEvent =
  | ({ type: "retrieval" } & Retrieval)
  | { type: "delta"; text: string }
  | { type: "done"; model?: string; cost_usd?: number | null; latency_ms: number }
  | { type: "error"; stage: "search" | "answer"; message: string };

export const CATEGORY_LABELS: Record<Category, string> = {
  quran: "القرآن",
  hadith: "الحديث",
  tafsir: "التفسير",
  fiqh: "الفقه",
  fiqh_usul: "أصول الفقه",
  aqeedah: "العقيدة",
  seerah: "السيرة",
};
