"use client";

import { Fragment, type ReactNode } from "react";
import { usePassages } from "@/lib/passages";

/**
 * A small markdown subset (paragraphs, "- " lists, **bold**) where [n] markers
 * become citation chips that open passage n.
 */
export function CitedText({ text, className }: { text: string; className?: string }) {
  const blocks = text.trim().split(/\n{2,}/);
  return (
    <div className={className}>
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        if (lines.every((l) => /^\s*[-•]\s+/.test(l))) {
          return (
            <ul key={i} className="cited-list">
              {lines.map((l, j) => (
                <li key={j}>
                  <Inline text={l.replace(/^\s*[-•]\s+/, "")} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i} dir="auto">
            <Inline text={lines.join(" ")} />
          </p>
        );
      })}
    </div>
  );
}

function Inline({ text }: { text: string }) {
  const out: ReactNode[] = [];
  // Citation runs like [1][3] or [1، 3] or [1, 3], and **bold**.
  const re = /\s*((?:\[\s*\d+(?:\s*[،,]\s*\d+)*\s*\])+)|\*\*(.+?)\*\*/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) {
      const nums = [...m[1].matchAll(/\d+/g)].map((d) => Number(d[0]));
      out.push(<CiteChips key={m.index} nums={nums} />);
    } else {
      out.push(<strong key={m.index}>{m[2]}</strong>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out}</>;
}

function CiteChips({ nums }: { nums: number[] }) {
  return (
    <span className="cite-run">
      {nums.map((n, i) => (
        <Fragment key={`${n}-${i}`}>
          <CiteChip n={n} />
        </Fragment>
      ))}
    </span>
  );
}

export function CiteChip({ n }: { n: number }) {
  const { passages, open } = usePassages();
  const p = passages.find((x) => x.n === n);
  if (!p) {
    return (
      <span className="cite-chip cite-chip--missing" title="لا يوجد مصدر بهذا الرقم">
        {n}
      </span>
    );
  }
  return (
    <button type="button" className="cite-chip" title={p.citation ?? undefined} onClick={() => open(n)}>
      {n}
    </button>
  );
}
