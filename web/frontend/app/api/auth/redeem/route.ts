/** POST /api/auth/redeem {code, name} -> the API's /auth/redeem; the token goes in a cookie. */
import { API, setToken } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const r = await fetch(`${API}/auth/redeem`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: await req.text(),
  });
  const data = await r.json();
  if (!r.ok) return Response.json(data, { status: r.status });
  await setToken(data.token);
  return Response.json({ user: data.user });
}
