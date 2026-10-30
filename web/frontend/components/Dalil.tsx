"use client";

import { Renderer, type ActionEvent } from "@openuidev/react-lang";
import { Button, Skeleton, ThemeProvider, createTheme } from "@openuidev/react-ui";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { library } from "@/lib/library";
import { PassagesContext } from "@/lib/passages";
import { CATEGORY_LABELS, type AskEvent, type Category, type Passage, type Retrieval } from "@/lib/types";
import { SourceDrawer } from "./SourceDrawer";
import { SourceList } from "./SourceList";

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
  question: string;
  categories: Category[];
  phase: "searching" | "answering" | "done" | "error";
  retrieval?: Retrieval;
  answer: string;
  error?: string;
  model?: string;
  cost?: number | null;
  totalMs?: number;
  rating?: string;
}

const EXAMPLES = [
  { q: "ما حكم المسح على الخفين وما شروطه؟", hint: "فقه · أقوال وأدلة" },
  { q: "إنما الأعمال بالنيات", hint: "حديث · تخريج ونص" },
  { q: "ما تفسير قوله تعالى: إن مع العسر يسرا؟", hint: "تفسير" },
  { q: "حلفت بالطلاق إن فعلت كذا ثم فعلته، فما الحكم؟", hint: "مسألة شخصية · إحالة" },
];

const CATEGORIES = Object.keys(CATEGORY_LABELS) as Category[];

interface User {
  id: number;
  name: string;
  role: string;
}

export function Dalil() {
  // undefined while checking the session, null when signed out.
  const [user, setUser] = useState<User | null | undefined>(undefined);
  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => (r.ok ? r.json() : null))
      .then(setUser, () => setUser(null));
  }, []);
  const signOut = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    setUser(null);
  };

  return (
    <ThemeProvider mode="light" lightTheme={theme}>
      {user === undefined ? null : user === null ? (
        <SignIn onSignedIn={setUser} />
      ) : (
        <Research user={user} onSignOut={signOut} onSignedOut={() => setUser(null)} />
      )}
    </ThemeProvider>
  );
}

function SignIn({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [code, setCode] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    const r = await fetch("/api/auth/redeem", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ code: code.trim(), name: name.trim() }),
    });
    setBusy(false);
    if (r.ok) onSignedIn((await r.json()).user);
    else setError(r.status === 403 ? "رمز الدعوة غير صالح أو مستعمل." : "تعذّر الدخول، حاول مرة أخرى.");
  };
  return (
    <main className="signin">
      <h1 className="welcome-title">دليل</h1>
      <p className="welcome-sub">الدخول برمز دعوة.</p>
      <form className="signin-form" onSubmit={submit}>
        <label>
          <span>الاسم</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} autoComplete="name" />
        </label>
        <label>
          <span>رمز الدعوة</span>
          <input value={code} onChange={(e) => setCode(e.target.value)} required dir="ltr" autoComplete="off" />
        </label>
        {error ? <p className="signin-error">{error}</p> : null}
        <Button variant="primary" size="medium" type="submit" disabled={busy || !code.trim() || !name.trim()}>
          {busy ? "جارٍ…" : "دخول"}
        </Button>
      </form>
    </main>
  );
}

function Research({ user, onSignOut, onSignedOut }: { user: User; onSignOut: () => void; onSignedOut: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [drawer, setDrawer] = useState<{ turnId: number; n: number } | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const busy = turns.some((t) => t.phase === "searching" || t.phase === "answering");
  const endRef = useRef<HTMLDivElement>(null);

  const patch = useCallback((id: number, f: (t: Turn) => Partial<Turn>) => {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...f(t) } : t)));
  }, []);

  const ask = useCallback(
    async (question: string, cats: Category[] = categories) => {
      const q = question.trim();
      if (!q) return;
      const id = Date.now();
      setTurns((ts) => [...ts, { id, question: q, categories: cats, phase: "searching", answer: "" }]);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }), 50);
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question: q, categories: cats }),
        });
        if (res.status === 401) return onSignedOut();
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
            if (e.type === "retrieval") {
              const { type: _t, ...retrieval } = e;
              patch(id, () => ({ phase: "answering", retrieval }));
            } else if (e.type === "delta") {
              patch(id, (t) => ({ answer: t.answer + e.text }));
            } else if (e.type === "done") {
              patch(id, () => ({ phase: "done", model: e.model, cost: e.cost_usd, totalMs: e.latency_ms }));
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
    [categories, patch, onSignedOut],
  );

  const drawerTurn = drawer ? turns.find((t) => t.id === drawer.turnId) : undefined;
  const drawerPassage = drawerTurn?.retrieval?.passages.find((p) => p.n === drawer?.n);

  return (
    <div className={`shell ${drawerPassage ? "has-drawer" : ""}`}>
      <header className="topbar">
        <button type="button" className="wordmark" onClick={() => setTurns([])} aria-label="دليل — البداية">
          دليل
        </button>
        <span className="topbar-sub">بحث موثّق في كتب التراث</span>
        <span className="topbar-spacer" />
        <span className="topbar-user">{user.name}</span>
        <button type="button" className="link-btn topbar-out" onClick={onSignOut}>
          خروج
        </button>
      </header>

      <main className="thread">
        <ServiceNotice />
        {turns.length === 0 ? (
          <Welcome onAsk={(q) => ask(q)} />
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

      <Composer busy={busy} categories={categories} setCategories={setCategories} onAsk={(q) => ask(q)} />

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
    : "قاعدة البيانات قيد التحميل، وسيعمل البحث بعد دقائق.";
  return <div className={`notice ${state === "offline" || state === "failed" ? "is-down" : ""}`}>{text}</div>;
}

/** Seconds since `from`, updated as it runs. */
function Elapsed({ from }: { from: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 200);
    return () => clearInterval(t);
  }, []);
  return <span className="elapsed">{fmtSeconds(now - from)}</span>;
}

function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <section className="welcome">
      <h1 className="welcome-title">دليل</h1>
      <p className="welcome-sub">
        اسأل بالعربية أو الإنجليزية، وتأتيك الإجابة من نصوص ٥٤٢ كتابًا في القرآن والحديث والتفسير والفقه والعقيدة
        والسيرة، وكل جملة فيها موثّقة بمصدرها.
      </p>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e.q} type="button" className="example" onClick={() => onAsk(e.q)}>
            <span className="example-q">{e.q}</span>
            <span className="example-hint">{e.hint}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function citedNumbers(answer: string): Set<number> {
  const nums = new Set<number>();
  for (const m of answer.matchAll(/\[(\d+)\]/g)) nums.add(Number(m[1]));
  for (const m of answer.matchAll(/Quote\(\s*(\d+)/g)) nums.add(Number(m[1]));
  for (const m of answer.matchAll(/View\([^\n]*\[([\d,\s]+)\]\)/g))
    for (const d of m[1].matchAll(/\d+/g)) nums.add(Number(d[0]));
  return nums;
}

const fmtSeconds = (ms: number) => `${(ms / 1000).toLocaleString("ar-EG", { maximumFractionDigits: 1 })} ث`;

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
      <div className="turn-meta">
        {turn.phase === "searching" || streaming ? (
          <span className="progress">
            <span className={turn.phase === "searching" ? "is-now" : "is-done"}>
              {turn.phase === "searching" ? "البحث في المصادر" : `${passages.length.toLocaleString("ar-EG")} مصادر`}
            </span>
            <span className="progress-sep">←</span>
            <span className={streaming ? "is-now" : ""}>كتابة الإجابة</span>
            <Elapsed from={turn.id} />
          </span>
        ) : turn.retrieval ? (
          <>
            <span>{passages.length.toLocaleString("ar-EG")} مصادر</span>
            {turn.totalMs ? (
              <>
                <span>·</span>
                <span>{fmtSeconds(turn.totalMs)}</span>
              </>
            ) : null}
            {turn.categories.length ? (
              <>
                <span>·</span>
                <span>{turn.categories.map((c) => CATEGORY_LABELS[c]).join("، ")}</span>
              </>
            ) : null}
          </>
        ) : null}
        {turn.retrieval?.status.retrieval?.startsWith("bm25_fallback") ? (
          <span className="meta-warn">بحث لفظي فقط: تعذّر البحث بالمعنى</span>
        ) : null}
      </div>

      <PassagesContext.Provider value={ctx}>
        {turn.phase === "searching" || (streaming && !turn.answer) ? (
          <div className="answer-skeleton">
            <Skeleton count={3} height="14px" />
          </div>
        ) : null}
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
      <span className="turn-cost" dir="ltr">
        {turn.model?.replace(/^anthropic\//, "")}
        {turn.cost != null ? ` · $${turn.cost.toFixed(3)}` : ""}
        {turn.totalMs ? ` · ${(turn.totalMs / 1000).toFixed(1)}s` : ""}
      </span>
    </footer>
  );
}

function Composer({
  busy, categories, setCategories, onAsk,
}: {
  busy: boolean;
  categories: Category[];
  setCategories: (c: Category[]) => void;
  onAsk: (q: string) => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
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
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={ref}
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
