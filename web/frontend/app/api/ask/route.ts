/**
 * POST /api/ask {question, categories?} -> NDJSON events (lib/types.ts AskEvent):
 * the retrieved passages first, then the answer as OpenUI Lang deltas.
 *
 * Retrieval is the Daleel API's /search. The answer step (any OpenAI-compatible
 * endpoint, see ANSWER_BASE_URL) runs here rather than
 * in the API's /ask because its system prompt is generated from the frontend's
 * component library (lib/system-prompt.generated.ts) and it streams.
 */
import { SYSTEM_PROMPT } from "@/lib/system-prompt.generated";
import type { AskEvent, Category, Passage } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API = process.env.DALEEL_API_URL ?? "http://localhost:8000";
const TOKEN = process.env.DALEEL_TOKEN ?? "";
// Any OpenAI-compatible chat endpoint. Default: DeepSeek 4.1 Flash on Ollama Cloud,
// the model Hermes runs; OpenRouter when ANSWER_BASE_URL points there.
const ANSWER_BASE_URL = process.env.ANSWER_BASE_URL ?? "https://ollama.com/v1";
const ANSWER_API_KEY = process.env.ANSWER_API_KEY ?? "";
const MODEL = process.env.ANSWER_MODEL ?? "deepseek-v4.1-flash:1017";
// DeepSeek reasons by default, and with a 3000-token cap it spent the whole budget
// reasoning and returned no answer. "none" measured 3.7 s per answer, "low" 7.8 s,
// both with every quote verified on four test questions. Empty sends no setting.
const REASONING_EFFORT = process.env.ANSWER_REASONING_EFFORT ?? "none";
const N_RESULTS = 8;
const PASSAGE_MAX_CHARS = 3000;

/** The same layout as the API's answer.render_user_message, so the model sees what /ask would show it. */
function userMessage(question: string, passages: Passage[]): string {
  const blocks = passages.map((p) => {
    let head = `[${p.n}] ${p.citation ?? p.parent_id}`;
    if (p.grade) head += ` (حكم الألباني: ${p.grade})`;
    let text = p.text;
    for (const nb of p.neighbours ?? []) {
      const where = (nb.offset ?? 0) < 0 ? "قبله" : "بعده";
      text += `\n\n(نص متصل ${where}: ${nb.citation ?? ""})\n${nb.text ?? ""}`;
    }
    return `${head}\n${text}`;
  });
  return `السؤال: ${question}\n\nالنصوص:\n\n${blocks.join("\n\n---\n\n")}`;
}

export async function POST(req: Request) {
  const body = (await req.json()) as { question?: string; categories?: Category[] };
  const question = (body.question ?? "").trim();
  if (question.length < 2) return Response.json({ error: "question is required" }, { status: 400 });

  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (e: AskEvent) => controller.enqueue(enc.encode(JSON.stringify(e) + "\n"));
      const started = Date.now();
      try {
        // 1. Retrieval
        const sr = await fetch(`${API}/search`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}` },
          body: JSON.stringify({
            question,
            n_results: N_RESULTS,
            categories: body.categories?.length ? body.categories : null,
            expand_neighbours: 1,
            passage_max_chars: PASSAGE_MAX_CHARS,
          }),
          signal: req.signal,
        });
        if (!sr.ok) {
          const detail = await sr.text();
          send({ type: "error", stage: "search", message: `Daleel API ${sr.status}: ${detail.slice(0, 300)}` });
          return;
        }
        const found = (await sr.json()) as {
          query_id: number; status: Record<string, string>; latency_ms: number; results: Omit<Passage, "n">[];
        };
        const passages: Passage[] = found.results.map((r, i) => ({ ...r, n: i + 1 }));
        send({ type: "retrieval", query_id: found.query_id, status: found.status, latency_ms: found.latency_ms, passages });

        // 2. The answer, streamed as OpenUI Lang
        const ar = await fetch(`${ANSWER_BASE_URL}/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${ANSWER_API_KEY}` },
          body: JSON.stringify({
            model: MODEL,
            stream: true,
            temperature: 0.2,
            max_tokens: 6000,
            ...(REASONING_EFFORT ? { reasoning_effort: REASONING_EFFORT } : {}),
            usage: { include: true }, // OpenRouter: report the call's cost
            messages: [
              { role: "system", content: SYSTEM_PROMPT },
              { role: "user", content: userMessage(question, passages) },
            ],
          }),
          signal: req.signal,
        });
        if (!ar.ok || !ar.body) {
          send({ type: "error", stage: "answer", message: `answer model HTTP ${ar.status}: ${(await ar.text()).slice(0, 300)}` });
          return;
        }
        let model: string | undefined;
        let cost: number | null = null;
        let finish: string | undefined;
        let wrote = false;
        const reader = ar.body.pipeThrough(new TextDecoderStream()).getReader();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += value;
          let nl: number;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue; // SSE comments (": OPENROUTER PROCESSING") and blanks
            const data = line.slice(5).trim();
            if (data === "[DONE]") continue;
            const chunk = JSON.parse(data);
            model ??= chunk.model;
            if (chunk.usage?.cost != null) cost = chunk.usage.cost;
            finish = chunk.choices?.[0]?.finish_reason ?? finish;
            const delta: string | undefined = chunk.choices?.[0]?.delta?.content;
            if (delta) {
              wrote = true;
              send({ type: "delta", text: delta });
            }
          }
        }
        if (!wrote || finish === "length") {
          send({
            type: "error", stage: "answer",
            message: wrote ? "The answer was cut off at the token limit." : "The model returned no answer text.",
          });
          return;
        }
        send({ type: "done", model, cost_usd: cost, latency_ms: Date.now() - started });
      } catch (err) {
        if (!req.signal.aborted) send({ type: "error", stage: "answer", message: String(err) });
      } finally {
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" },
  });
}
