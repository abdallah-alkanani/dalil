"use client";

import { CATEGORY_LABELS, type Passage } from "@/lib/types";

/** Every retrieved passage, numbered; the ones the answer cites come first and are marked. */
export function SourceList({
  passages, cited, activeN, onOpen,
}: {
  passages: Passage[];
  cited: Set<number>;
  activeN?: number;
  onOpen: (n: number) => void;
}) {
  const ordered = [...passages].sort((a, b) => Number(cited.has(b.n)) - Number(cited.has(a.n)) || a.n - b.n);
  const usedCount = passages.filter((p) => cited.has(p.n)).length;
  return (
    <section className="sources">
      <div className="sources-head">
        <h3>المصادر</h3>
        <span>
          {usedCount.toLocaleString("ar-EG")} مستشهد بها من {passages.length.toLocaleString("ar-EG")}
        </span>
      </div>
      <ol className="sources-grid">
        {ordered.map((p) => {
          const [book, ...rest] = (p.citation ?? p.parent_id).split("، ");
          return (
            <li key={p.parent_id}>
              <button
                type="button"
                className={`source-card ${cited.has(p.n) ? "is-cited" : ""} ${activeN === p.n ? "is-active" : ""}`}
                onClick={() => onOpen(p.n)}
              >
                <span className="source-n">{p.n}</span>
                <span className="source-body">
                  <span className="source-book">{book}</span>
                  {rest.length ? <span className="source-sec">{rest.join("، ")}</span> : null}
                  <span className="source-tags">
                    {p.category ? <span className={`cat-dot cat-${p.category}`}>{CATEGORY_LABELS[p.category]}</span> : null}
                    {p.grade ? <span className="grade-mini">{p.grade}</span> : null}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
