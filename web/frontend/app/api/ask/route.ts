/**
 * POST /api/ask {question, categories?} -> the API's streaming /ask, as the
 * signed-in visitor: NDJSON events (lib/types.ts AskEvent), passed through.
 */
import { api } from "@/lib/session";
import type { AskEvent } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const r = await api("/ask", { method: "POST", body: await req.text(), signal: req.signal });
  if (r.ok && r.body) {
    return new Response(r.body, {
      headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" },
    });
  }
  // Refused before the stream started (not signed in, a daily cap, search not ready).
  const detail = await r.json().then((d) => d.detail ?? d, () => r.statusText);
  const event: AskEvent = { type: "error", stage: "search", message: typeof detail === "string" ? detail : JSON.stringify(detail) };
  return new Response(JSON.stringify(event) + "\n", {
    status: r.status,
    headers: { "content-type": "application/x-ndjson; charset=utf-8" },
  });
}
