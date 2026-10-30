/** GET /api/health -> the API's /health and its search state. */
import { API } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    return Response.json(await r.json(), { status: r.status });
  } catch {
    return Response.json({ ok: false, search: "offline" }, { status: 503 });
  }
}
