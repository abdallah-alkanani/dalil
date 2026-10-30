/** POST /api/auth/logout: forget the session cookie. */
import { clearToken } from "@/lib/session";

export const runtime = "nodejs";

export async function POST() {
  await clearToken();
  return Response.json({ ok: true });
}
