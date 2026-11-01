/**
 * POST /api/ask {question, categories?} -> the API's streaming /ask for this
 * visitor: NDJSON events (lib/types.ts AskEvent), passed through.
 */
import { api } from "@/lib/session";
import { DAILY_LIMIT_TEXT, SERVICE_TEXT } from "@/lib/service";
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
  // Refused before the stream started: the API is unreachable, search is not ready, or the daily cap.
  const body = await r.json().catch(() => ({}));
  const detail = typeof body.detail === "string" ? body.detail : "";
  const event: AskEvent =
    r.status === 429 ? { type: "error", stage: "limit", message: DAILY_LIMIT_TEXT }
    : r.status === 503 ? { type: "error", stage: "service", message: SERVICE_TEXT[unavailable(body.service, detail)] }
    : { type: "error", stage: "search", message: detail || r.statusText };
  return new Response(JSON.stringify(event) + "\n", {
    status: r.status,
    headers: { "content-type": "application/x-ndjson; charset=utf-8" },
  });
}

/** Why the API refused with 503: unreachable, its stores failed, or still loading them. */
function unavailable(service: string | undefined, detail: string): "unreachable" | "failed" | "loading" {
  if (service === "unreachable") return "unreachable";
  return /\(failed/.test(detail) ? "failed" : "loading";
}
