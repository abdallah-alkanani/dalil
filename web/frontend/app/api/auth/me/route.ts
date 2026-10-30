/** GET /api/auth/me -> the signed-in user, or null when signed out. */
import { api, token } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  if (!(await token())) return Response.json(null);
  const r = await api("/me");
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
