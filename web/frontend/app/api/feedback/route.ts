/** POST /api/feedback -> the Daleel API's /feedback, with the server's token. */
export const runtime = "nodejs";

const API = process.env.DALEEL_API_URL ?? "http://localhost:8000";
const TOKEN = process.env.DALEEL_TOKEN ?? "";

export async function POST(req: Request) {
  const r = await fetch(`${API}/feedback`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}` },
    body: await req.text(),
  });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
