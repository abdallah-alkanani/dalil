/**
 * The site is open to everyone, with no sign-in. Each browser gets a random visitor
 * id in an httpOnly cookie, and every API call carries it with SITE_TOKEN, so the
 * API can tell visitors apart and apply its per-visitor caps. The call also carries
 * the country the host reports or, without one, the visitor's IP, which the API turns
 * into a country and does not keep.
 */
import { cookies, headers } from "next/headers";

export const API = process.env.DALIL_API_URL ?? "http://localhost:8000";
const COOKIE = "dalil_visitor";

async function visitorId(): Promise<string> {
  const jar = await cookies();
  const known = jar.get(COOKIE)?.value;
  if (known) return known;
  const id = crypto.randomUUID();
  jar.set(COOKIE, id, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 365,
  });
  return id;
}

/** Where the visitor is: the host's country header (Vercel, Cloudflare), or else their IP. */
async function whereFrom(): Promise<Record<string, string>> {
  const h = await headers();
  const country = h.get("x-vercel-ip-country") ?? h.get("cf-ipcountry");
  if (country) return { "x-dalil-country": country };
  const ip = h.get("x-forwarded-for")?.split(",")[0].trim() || h.get("x-real-ip");
  return ip ? { "x-dalil-ip": ip } : {};
}

/** A call to the API for this visitor. */
export async function api(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${API}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.SITE_TOKEN ?? ""}`,
      "x-dalil-visitor": await visitorId(),
      ...(await whereFrom()),
      ...init.headers,
    },
  });
}
