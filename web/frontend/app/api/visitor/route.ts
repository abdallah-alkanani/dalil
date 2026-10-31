/** GET /api/visitor -> this visitor's name; POST {name} -> the name they choose to give. */
import { api } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const r = await api("/visitor");
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}

export async function POST(req: Request) {
  const r = await api("/visitor", { method: "POST", body: await req.text() });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
