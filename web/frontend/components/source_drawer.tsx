"use client";

import { IconButton } from "@openuidev/react-ui";
import { useEffect, useState } from "react";
import { CATEGORY_LABELS, type Passage } from "@/lib/types";

const RELEVANCE = [
  { value: "direct", label: "يجيب مباشرة" },
  { value: "supporting", label: "داعم" },
  { value: "topical", label: "في الموضوع" },
  { value: "irrelevant", label: "غير ذي صلة" },
] as const;

/** One passage in full, with the reader's relevance judgment (POST /feedback). */
export function SourceDrawer({
  passage: p, queryId, total, onNavigate, onClose,
}: {
  passage: Passage;
  queryId: number;
  total: number;
  onNavigate: (n: number) => void;
  onClose: () => void;
}) {
  const [judged, setJudged] = useState<Record<string, string>>({});
  const [showNeighbours, setShowNeighbours] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      // RTL: the left arrow moves forward.
      if (e.key === "ArrowLeft" && p.n < total) onNavigate(p.n + 1);
      if (e.key === "ArrowRight" && p.n > 1) onNavigate(p.n - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [p.n, total, onNavigate, onClose]);

  useEffect(() => setShowNeighbours(false), [p.parent_id]);

  const judge = async (relevance: string) => {
    const r = await fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query_id: queryId, passages: [{ parent_id: p.parent_id, relevance }] }),
    });
    if (r.ok) setJudged((j) => ({ ...j, [p.parent_id]: relevance }));
  };

  const [book] = (p.citation ?? p.parent_id).split("، ");
  const hadith = p.refs?.hadith;
  const quran = p.refs?.quran;
  const before = p.neighbours.filter((n) => n.offset < 0);
  const after = p.neighbours.filter((n) => n.offset > 0);

  return (
    <aside className="drawer" aria-label="نص المصدر">
      <div className="drawer-head">
        <span className="drawer-n">{p.n}</span>
        <div className="drawer-title">
          <div className="drawer-book">{book}</div>
          {p.section_path.length > 1 ? (
            <nav className="crumbs">
              {p.section_path.slice(1).map((s, i) => (
                <span key={i}>{s}</span>
              ))}
            </nav>
          ) : null}
        </div>
        <div className="drawer-nav">
          <IconButton
            variant="tertiary" size="small" icon={<span aria-hidden>›</span>} aria-label="السابق"
            disabled={p.n <= 1} onClick={() => onNavigate(p.n - 1)}
          />
          <IconButton
            variant="tertiary" size="small" icon={<span aria-hidden>‹</span>} aria-label="التالي"
            disabled={p.n >= total} onClick={() => onNavigate(p.n + 1)}
          />
          <IconButton variant="tertiary" size="small" icon={<span aria-hidden>✕</span>} aria-label="إغلاق" onClick={onClose} />
        </div>
      </div>

      <div className="drawer-meta">
        {[
          p.category ? CATEGORY_LABELS[p.category] : null,
          hadith?.hadith_number ? `حديث رقم ${hadith.hadith_number}` : null,
          quran?.surah ? `سورة ${quran.surah}، آية ${quran.ayah ?? ""}` : null,
          p.page_number ? `ص ${p.page_number}` : null,
          p.grade ? `حكم الألباني: ${p.grade}` : null,
        ].filter(Boolean).join(" · ")}
      </div>

      <div className="drawer-scroll">
        {before.length && showNeighbours
          ? before.map((nb) => <Neighbour key={nb.parent_id} text={nb.text} citation={nb.citation} label="النص السابق" />)
          : null}
        <div className="passage-text">{p.text}</div>
        {p.truncated ? <div className="truncated">… النص مقتطع حول موضع المطابقة</div> : null}
        {after.length && showNeighbours
          ? after.map((nb) => <Neighbour key={nb.parent_id} text={nb.text} citation={nb.citation} label="النص اللاحق" />)
          : null}
        {p.neighbours.length ? (
          <button type="button" className="link-btn" onClick={() => setShowNeighbours((v) => !v)}>
            {showNeighbours ? "إخفاء السياق" : "عرض ما قبله وما بعده"}
          </button>
        ) : null}
      </div>

      <div className="drawer-judge">
        <span className="rate-label">هل يجيب هذا النص عن السؤال؟</span>
        <div className="judge-row">
          {RELEVANCE.map((r) => (
            <button
              key={r.value}
              type="button"
              className={`rate-btn ${judged[p.parent_id] === r.value ? "is-on" : ""}`}
              onClick={() => judge(r.value)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

function Neighbour({ text, citation, label }: { text: string; citation: string | null; label: string }) {
  return (
    <div className="neighbour">
      <div className="neighbour-label">
        {label}
        {citation ? ` — ${citation.split("، ").slice(1).join("، ")}` : ""}
      </div>
      <div className="passage-text is-muted">{text}</div>
    </div>
  );
}
