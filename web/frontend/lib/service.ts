/**
 * Whether search can answer right now, and what to tell visitors when it cannot.
 * "unreachable": the site cannot reach the API at all. "failed": the API is up but
 * its search stores are not (for example, the data volume is missing). "loading":
 * the API is still downloading, building or opening the stores.
 */
export type ServiceState = "ready" | "loading" | "failed" | "unreachable";

export const SERVICE_TEXT: Record<Exclude<ServiceState, "ready">, string> = {
  unreachable: "خدمة البحث متوقفة الآن، فلا يمكن الإجابة عن الأسئلة. إعادة المحاولة لن تفيد حتى تعود الخدمة.",
  failed: "قاعدة الكتب غير متاحة لخدمة البحث الآن، فالبحث متوقف حتى تعود.",
  loading: "خدمة البحث تُحمِّل قاعدة الكتب، والبحث متوقف حتى يكتمل التحميل.",
};

export const DAILY_LIMIT_TEXT = "بلغ الموقع حدّ الأسئلة لهذا اليوم، ويُفتح من جديد عند منتصف الليل بتوقيت غرينتش.";
export const SITE_UNREACHABLE_TEXT = "تعذّر الوصول إلى الموقع. تحقّق من اتصالك بالإنترنت ثم أعد المحاولة.";

/** The service state from the API's /health search field (or "unreachable"). */
export function serviceState(search: string | undefined): ServiceState {
  if (search === "ready") return "ready";
  if (search === "failed") return "failed";
  if (search === "unreachable" || !search) return "unreachable";
  return "loading";
}
