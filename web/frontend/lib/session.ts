/**
 * Each visitor signs in with an invite code; the API's bearer token is kept in an
 * httpOnly cookie and sent with every call, so the API's per-user caps apply.
 */
import { cookies } from "next/headers";

export const API = process.env.DALIL_API_URL ?? "http://localhost:8000";
const COOKIE = "dalil_token";

export async function token(): Promise<string | undefined> {
  return (await cookies()).get(COOKIE)?.value;
}

export async function setToken(value: string): Promise<void> {
  (await cookies()).set(COOKIE, value, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 365,
  });
}

export async function clearToken(): Promise<void> {
  (await cookies()).delete(COOKIE);
}

/** A call to the API as the signed-in visitor; 401 without a session. */
export async function api(path: string, init: RequestInit = {}): Promise<Response> {
  const t = await token();
  if (!t) return Response.json({ detail: "Not signed in" }, { status: 401 });
  return fetch(`${API}${path}`, {
    ...init,
    headers: { "content-type": "application/json", authorization: `Bearer ${t}`, ...init.headers },
  });
}
