"use client";

import { Renderer, type ActionEvent } from "@openuidev/react-lang";
import { Button, ThemeProvider, createTheme } from "@openuidev/react-ui";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { library } from "@/lib/library";
import { PassagesContext } from "@/lib/passages";
import { CATEGORY_LABELS, type AskEvent, type Category, type Passage, type Retrieval } from "@/lib/types";
import { SourceDrawer } from "./source_drawer";
import { SourceList } from "./source_list";

const theme = createTheme({
  background: "#f7f5f0",
  foreground: "#ffffff",
  textNeutralPrimary: "#1d1c1a",
  textNeutralSecondary: "#57534e",
  textNeutralTertiary: "#8a857d",
  textBrand: "#1e5b4f",
  textAccentPrimary: "#1e5b4f",
  interactiveAccentDefault: "#1e5b4f",
  interactiveAccentHover: "#174a40",
  interactiveAccentPressed: "#123b33",
  interactiveAccentDisabled: "#b9cdc6",
  borderDefault: "#e7e3da",
  fontBody: "var(--font-sans)",
  fontHeading: "var(--font-sans)",
  fontLabel: "var(--font-sans)",
  fontNumbers: "var(--font-sans)",
});

interface Turn {
  id: number;
  startedAt: number;
  question: string;
  categories: Category[];
  phase: "searching" | "answering" | "done" | "error";
  retrieval?: Retrieval;
  answer: string;
  error?: string;
  rating?: string;
}

const CATEGORIES = Object.keys(CATEGORY_LABELS) as Category[];

export function Dalil() {
  return (
    <ThemeProvider mode="light" lightTheme={theme}>
      <Research />
    </ThemeProvider>
  );
}

function Research() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [drawer, setDrawer] = useState<{ turnId: number; n: number } | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const busy = turns.some((t) => t.phase === "searching" || t.phase === "answering");
  const endRef = useRef<HTMLDivElement>(null);
  const turnSeq = useRef(0);

  const patch = useCallback((id: number, f: (t: Turn) => Partial<Turn>) => {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...f(t) } : t)));
  }, []);

  const ask = useCallback(
    async (question: string, cats: Category[] = categories) => {
      const q = question.trim();
      if (!q) return;
      const id = (turnSeq.current += 1);
      const turn: Turn = { id, startedAt: Date.now(), question: q, categories: cats, phase: "searching", answer: "" };
      setTurns((ts) => [...ts, turn]);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }), 50);
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question: q, categories: cats }),
        });
        if (!res.body) throw new Error(`HTTP ${res.status}`);
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += value;
          let nl: number;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl);
            buf = buf.slice(nl + 1);
            if (!line.trim()) continue;
            const e = JSON.parse(line) as AskEvent;
            if (e.type === "start") {
              patch(id, () => ({ retrieval: { query_id: e.query_id, searches: [], passages: [] } }));
            } else if (e.type === "search") {
              patch(id, (t) => ({
                retrieval: t.retrieval && {
                  ...t.retrieval,
                  searches: [...t.retrieval.searches, { query: e.query, count: e.passages.length, status: e.status }],
                  passages: [...t.retrieval.passages, ...e.passages],
                },
              }));
            } else if (e.type === "reset") {
              patch(id, () => ({ phase: "searching", answer: "" }));
            } else if (e.type === "delta") {
              patch(id, (t) => ({ phase: "answering", answer: t.answer + e.text }));
            } else if (e.type === "done") {
              patch(id, () => ({ phase: "done" }));
            } else if (e.type === "error") {
              patch(id, () => ({ phase: "error", error: e.message }));
            }
          }
        }
        patch(id, (t) => (t.phase === "answering" || t.phase === "searching" ? { phase: "done" } : {}));
      } catch (err) {
        patch(id, () => ({ phase: "error", error: String(err) }));
      }
    },
    [categories, patch],
  );

  const drawerTurn = drawer ? turns.find((t) => t.id === drawer.turnId) : undefined;
  const drawerPassage = drawerTurn?.retrieval?.passages.find((p) => p.n === drawer?.n);

  return (
    <div className={`shell ${drawerPassage ? "has-drawer" : ""}`}>
      <header className="topbar">
        <button type="button" className="wordmark" onClick={() => setTurns([])} aria-label="دليل — البداية">
          دليل
        </button>
        <span className="topbar-sub">من القرآن والسنة وأقوال أهل العلم</span>
        <span className="topbar-spacer" />
        <VisitorName />
      </header>

      <main className={`thread ${turns.length ? "" : "is-empty"}`}>
        <ServiceNotice />
        {turns.length === 0 ? (
          <Welcome>
            <Composer busy={busy} categories={categories} setCategories={setCategories} onAsk={(q) => ask(q)} centered />
          </Welcome>
        ) : (
          turns.map((t) => (
            <TurnView
              key={t.id}
              turn={t}
              activeN={drawer?.turnId === t.id ? drawer.n : undefined}
              onOpen={(n) => setDrawer({ turnId: t.id, n })}
              onFollowUp={(q) => ask(q, t.categories)}
              onRated={(rating) => patch(t.id, () => ({ rating }))}
              busy={busy}
            />
          ))
        )}
        <div ref={endRef} />
      </main>

      {turns.length ? (
        <Composer busy={busy} categories={categories} setCategories={setCategories} onAsk={(q) => ask(q)} />
      ) : null}

      {drawerPassage && drawerTurn?.retrieval ? (
        <SourceDrawer
          passage={drawerPassage}
          queryId={drawerTurn.retrieval.query_id}
          total={drawerTurn.retrieval.passages.length}
          onNavigate={(n) => setDrawer({ turnId: drawerTurn.id, n })}
          onClose={() => setDrawer(null)}
        />
      ) : null}
    </div>
  );
}

/** Says something only when search is not available yet or not at all. */
function ServiceNotice() {
  const [state, setState] = useState<string>("ready");
  useEffect(() => {
    let alive = true;
    const tick = () =>
      fetch("/api/health")
        .then((r) => r.json())
        .then((d) => alive && setState(d.search ?? "unknown"))
        .catch(() => alive && setState("offline"));
    tick();
    const t = setInterval(tick, 15000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);
  if (state === "ready") return null;
  const text =
    state === "offline" ? "تعذّر الاتصال بالخادم. أعد المحاولة بعد قليل."
    : state === "failed" ? "تعذّر تحميل قاعدة البيانات، فالبحث متوقف الآن."
    : "قاعدة البيانات قيد التحميل، والبحث متوقف حتى يكتمل.";
  return <div className={`notice ${state === "offline" || state === "failed" ? "is-down" : ""}`}>{text}</div>;
}

/**
 * The name the visitor chooses to give, so the people who run the site know who is
 * asking. Optional: without it the visitor stays anonymous.
 */
function VisitorName() {
  const [name, setName] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    fetch("/api/visitor")
      .then((r) => (r.ok ? r.json() : null))
      .then((v) => setName(v?.name ?? ""), () => setName(""));
  }, []);
  if (name === null) return null;
  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    const r = await fetch("/api/visitor", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name: draft.trim() }),
    });
    if (r.ok) setName((await r.json()).name);
    setEditing(false);
  };
  if (editing) {
    return (
      <form className="visitor-form" onSubmit={save}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="اسمك (اختياري)"
          maxLength={200}
          autoFocus
          autoComplete="name"
        />
        <button type="submit" className="link-btn">حفظ</button>
      </form>
    );
  }
  return (
    <button
      type="button"
      className="link-btn visitor-name"
      title="اختياري: ليعرف القائمون على الموقع من يسأل"
      onClick={() => {
        setDraft(name);
        setEditing(true);
      }}
    >
      {name || "عرّفنا باسمك"}
    </button>
  );
}

function Welcome({ children }: { children: React.ReactNode }) {
  return (
    <section className="welcome">
      <h1 className="welcome-title">دليل</h1>
      <p className="welcome-sub">
        اسأل بالعربية أو الإنجليزية، وتأتيك الإجابة من نصوص ٥٤٢ كتابًا في القرآن والحديث والتفسير والفقه والعقيدة
        والسيرة، وكل جملة فيها موثّقة بمصدرها.
      </p>
      {children}
    </section>
  );
}

function citedNumbers(answer: string): Set<number> {
  const nums = new Set<number>();
  for (const m of answer.matchAll(/\[(\d+)\]/g)) nums.add(Number(m[1]));
  for (const m of answer.matchAll(/Quote\(\s*(\d+)/g)) nums.add(Number(m[1]));
  return nums;
}

function TurnView({
  turn, activeN, onOpen, onFollowUp, onRated, busy,
}: {
  turn: Turn;
  activeN?: number;
  onOpen: (n: number) => void;
  onFollowUp: (q: string) => void;
  onRated: (rating: string) => void;
  busy: boolean;
}) {
  const passages: Passage[] = turn.retrieval?.passages ?? [];
  const searches = turn.retrieval?.searches ?? [];
  const ctx = useMemo(() => ({ passages, open: onOpen }), [passages, onOpen]);
  const cited = useMemo(() => citedNumbers(turn.answer), [turn.answer]);
  const streaming = turn.phase === "answering";
  const onAction = useCallback(
    (e: ActionEvent) => {
      if (e.type === "continue_conversation" && !busy) onFollowUp(e.humanFriendlyMessage);
    },
    [busy, onFollowUp],
  );

  return (
    <section className="turn">
      <h2 className="turn-q" dir="auto">
        {turn.question}
      </h2>
      {turn.phase === "searching" || streaming ? (
        <Progress turn={turn} />
      ) : (
        <div className="turn-meta">
          {passages.length ? <span>{passages.length.toLocaleString("ar-EG")} مصادر</span> : null}
          {turn.categories.length ? (
            <>
              <span>·</span>
              <span>{turn.categories.map((c) => CATEGORY_LABELS[c]).join("، ")}</span>
            </>
          ) : null}
          {searches.some((x) => x.status.retrieval?.startsWith("bm25_fallback")) ? (
            <span className="meta-warn">تعذّر البحث الكامل هذه المرة، فقد تفوت الإجابة بعض النصوص</span>
          ) : null}
        </div>
      )}

      <PassagesContext.Provider value={ctx}>
        {turn.answer ? (
          <div className="answer">
            <Renderer
              response={turn.answer}
              library={library}
              isStreaming={streaming}
              onAction={onAction}
            />
          </div>
        ) : null}
        {turn.phase === "error" ? <div className="turn-error">تعذّر إكمال الطلب: {turn.error}</div> : null}
        {passages.length ? <SourceList passages={passages} cited={cited} activeN={activeN} onOpen={onOpen} /> : null}
      </PassagesContext.Provider>

      {turn.phase === "done" && turn.retrieval ? (
        <TurnFooter turn={turn} queryId={turn.retrieval.query_id} onRated={onRated} />
      ) : null}
    </section>
  );
}

/** Wording for each stage of an answer; within a stage it moves on every few seconds. */
const STAGES = {
  reading: ["أقرأ سؤالك", "أبحث في الكتب", "أقلّب الصفحات"],
  searched: (query: string) => [`أبحث عن «${query}»`, "أقرأ النصوص", "أجمع أقوال العلماء", "أرتّب الأدلة"],
  writing: ["أكتب الإجابة"],
};
const STAGE_STEP_MS = 2600;

/** A line that changes as the answer moves on: reading, searching, reading the texts, writing. */
function Progress({ turn }: { turn: Turn }) {
  const searches = turn.retrieval?.searches ?? [];
  const writing = turn.phase === "answering" && /(^|\n)\s*root\s*=/.test(turn.answer);
  const stage = writing ? "writing" : searches.length ? `searched-${searches.length}` : "reading";
  const lines = writing ? STAGES.writing
    : searches.length ? STAGES.searched(searches[searches.length - 1].query)
    : STAGES.reading;
  const [step, setStep] = useState(0);
  useEffect(() => {
    setStep(0);
    const t = setInterval(() => setStep((n) => n + 1), STAGE_STEP_MS);
    return () => clearInterval(t);
  }, [stage]);
  const text = lines[Math.min(step, lines.length - 1)];
  return (
    <div className="progress" role="status" aria-live="polite">
      <span className="progress-mark" aria-hidden />
      <span key={text} className="progress-text">{text}…</span>
    </div>
  );
}

const RATINGS: { value: string; label: string }[] = [
  { value: "correct", label: "صحيحة" },
  { value: "partly_correct", label: "صحيحة جزئيًا" },
  { value: "incorrect", label: "خاطئة" },
  { value: "unsafe", label: "غير آمنة" },
];

function TurnFooter({ turn, queryId, onRated }: { turn: Turn; queryId: number; onRated: (r: string) => void }) {
  const rate = async (rating: string) => {
    const r = await fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query_id: queryId, rating }),
    });
    if (r.ok) onRated(rating);
  };
  return (
    <footer className="turn-footer">
      <div className="rate">
        <span className="rate-label">تقييم الإجابة</span>
        {RATINGS.map((r) => (
          <button
            key={r.value}
            type="button"
            className={`rate-btn ${turn.rating === r.value ? "is-on" : ""}`}
            onClick={() => rate(r.value)}
          >
            {r.label}
          </button>
        ))}
      </div>
    </footer>
  );
}

function Composer({
  busy, categories, setCategories, onAsk, centered = false,
}: {
  busy: boolean;
  categories: Category[];
  setCategories: (c: Category[]) => void;
  onAsk: (q: string) => void;
  /** In the middle of the start page instead of fixed at the bottom. */
  centered?: boolean;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  // Safari on iPhone never shrinks the page for the keyboard; only the visible area
  // shrinks, so a box fixed to the bottom ends up behind it. Lift it by what the
  // keyboard covers. Where the page does shrink, this measures 0.
  useEffect(() => {
    const vv = window.visualViewport;
    if (centered || !vv) return;
    const root = document.documentElement;
    const place = () =>
      root.style.setProperty("--keyboard-inset", `${Math.max(0, window.innerHeight - vv.height - vv.offsetTop)}px`);
    place();
    vv.addEventListener("resize", place);
    vv.addEventListener("scroll", place);
    return () => {
      vv.removeEventListener("resize", place);
      vv.removeEventListener("scroll", place);
      root.style.removeProperty("--keyboard-inset");
    };
  }, [centered]);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, window.innerWidth < 640 ? 120 : 180)}px`;
  }, [text]);
  const submit = () => {
    if (busy || text.trim().length < 2) return;
    onAsk(text);
    setText("");
  };
  const toggle = (c: Category) =>
    setCategories(categories.includes(c) ? categories.filter((x) => x !== c) : [...categories, c]);

  return (
    <div className={`composer-wrap ${centered ? "is-centered" : ""}`}>
      <div className="composer">
        <textarea
          ref={ref}
          autoFocus={centered}
          dir="auto"
          rows={1}
          value={text}
          placeholder="اكتب سؤالك…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className="composer-row">
          <div className="cat-chips" role="group" aria-label="تصفية حسب الفن">
            <button
              type="button"
              className={`cat-chip ${categories.length === 0 ? "is-on" : ""}`}
              onClick={() => setCategories([])}
            >
              الكل
            </button>
            {CATEGORIES.map((c) => (
              <button
                key={c}
                type="button"
                className={`cat-chip ${categories.includes(c) ? "is-on" : ""}`}
                onClick={() => toggle(c)}
              >
                {CATEGORY_LABELS[c]}
              </button>
            ))}
          </div>
          <Button variant="primary" size="medium" disabled={busy || text.trim().length < 2} onClick={submit}>
            {busy ? "جارٍ…" : "اسأل"}
          </Button>
        </div>
      </div>
    </div>
  );
}
