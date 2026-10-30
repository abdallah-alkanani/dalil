/**
 * Finding a model's quotation in the passage it cites.
 *
 * Both sides are normalised the way readers treat them as equal: diacritics and
 * tatweel dropped, alef/yaa/taa-marbuta forms unified, punctuation ignored, and
 * whitespace collapsed. The match is mapped back to the original text, so what is
 * shown is the source's own wording, diacritics included, not the model's copy.
 */

const DIACRITICS = /[ؐ-ًؚ-ٰٟۖ-ۭـ]/;
const PUNCT = /[\p{P}\p{S}]/u;

function fold(ch: string): string {
  switch (ch) {
    case "أ": case "إ": case "آ": case "ٱ": return "ا";
    case "ى": return "ي";
    case "ة": return "ه";
    case "ؤ": return "و";
    case "ئ": return "ي";
    default: return ch.toLowerCase();
  }
}

/** The normalised text and, for each of its characters, the index it came from. */
function normalise(text: string): { norm: string; map: number[] } {
  let norm = "";
  const map: number[] = [];
  let pendingSpace = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (DIACRITICS.test(ch)) continue;
    if (/\s/.test(ch) || PUNCT.test(ch)) {
      pendingSpace = norm.length > 0;
      continue;
    }
    if (pendingSpace) {
      norm += " ";
      map.push(i);
      pendingSpace = false;
    }
    norm += fold(ch);
    map.push(i);
  }
  return { norm, map };
}

/** The [start, end) span of `quote` in `source`, or null when it is not there verbatim. */
export function findVerbatim(source: string, quote: string): [number, number] | null {
  const q = normalise(quote).norm.trim();
  if (q.length < 6) return null;
  const { norm, map } = normalise(source);
  const at = norm.indexOf(q);
  if (at < 0) return null;
  return [map[at], map[at + q.length - 1] + 1];
}
