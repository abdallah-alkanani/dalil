/** POST /api/feedback -> the API's /feedback for this visitor. */
import { api } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const r = await api("/feedback", { method: "POST", body: await req.text() });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
