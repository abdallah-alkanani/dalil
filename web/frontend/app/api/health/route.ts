/** GET /api/health -> the API's /health and its search state. */
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API = process.env.DALIL_API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    return Response.json(await r.json(), { status: r.status });
  } catch {
    return Response.json({ ok: false, search: "offline" }, { status: 503 });
  }
}
