/**
 * Writes web/backend/dalil_api/answer_components.md from the component library:
 * the OpenUI instructions the API appends to prompts/answering_system_prompt.md.
 *
 *     npm run prompt
 */
import { writeFileSync } from "node:fs";
import { library } from "../lib/library";

const preamble = `# On the website

On the website you search with the \`search\` tool, not the command line. Call it with an Arabic query; it returns numbered passages, and the numbers continue across searches. Search at most 3 times for one question: search first, and when a search misses, search again with other words. Every rule above still applies. Then write the answer as openui-lang with the components below; the interface shows the question, so start with the short answer.`;

const rules = [
  "root is always Answer([...]). Map the answer shape above onto it: الجواب المختصر is the Lead; the evidence and the views follow as Quote, Prose and Views blocks, Quran first, then hadith, then the scholars' words.",
  "Refer to a passage by its number. Put [n] after every statement that rests on passage n, and name who said it and in which book in the sentence itself.",
  "Quote pastes text from one passage: whole sentences, exactly as in the passage, with the part that answers the question in **bold**. The interface checks the words against the passage and shows the passage's own text.",
  "Views holds the views when the passages contain more than one. Each View names who holds it and in which book, and carries their own words as a Quote from the passage that states it.",
  "Your own connection of the evidence goes in a Prose whose heading is «استنتاج من النصوص أعلاه، وليس نصًا لأحد العلماء», with [n] for the passages it rests on.",
  "Gap says what the passages do not answer; when nothing answers the question, the answer is a Lead saying «لم أجد نصًا يجيب عن السؤال في قاعدة البيانات» and nothing else.",
  "Referral tells the asker to take a personal case to a qualified scholar.",
  "Steps only when a passage sets out a procedure; Table only to set short facts from several passages side by side.",
  "End with a FollowUpBlock of two or three follow-up questions in the user's language that the passages could answer.",
];

// Passage 1 is جامع تراث العلامة الألباني في الفقه, "مشروعية المسح على الخف أو الجورب المخرق";
// passage 2 is مجموع فتاوى ورسائل العثيمين, fatwa 105. The quotes are their text.
const example = `root = Answer([lead, views, fu])
lead = Lead("نقل الألباني في جامع تراث العلامة الألباني في الفقه أن أكثر العلماء يمنعون المسح على الخف المخرق، واختار هو جوازه [1]. ورجّح ابن عثيمين في مجموع فتاوى ورسائل العثيمين جواز المسح على الجورب المخرق [2].")
views = Views("المسح على الخف المخرق", [v1, v2])
v1 = View("الألباني في جامع تراث العلامة الألباني في الفقه", v1q)
v1q = Quote(1, "وأما المسح على الخف أو الجورب المخرق فقد اختلفوا فيه اختلافا كثيرا فأكثرهم يمنع منه على خلاف طويل بينهم تراه في مبسوطات الكتب الفقهية و «المحلى»، **وذهب غيرهم إلى الجواز وهو الذي نختاره**")
v2 = View("ابن عثيمين في مجموع فتاوى ورسائل العثيمين", v2q)
v2q = Quote(2, "فأجاب - حفظه الله تعالى - بقوله: **القول الراجح أنه يجوز المسح على الجورب المخرق والجورب الخفيف الذي تُرى من ورائه البشرة**، لأنه ليس المقصود من جواز المسح على الجورب ونحوه ان يكون ساتراً، فإن الرِّجل ليست عورة يجب سترها")
fu = FollowUpBlock([f1, f2])
f1 = FollowUpItem("ما شروط المسح على الخفين؟")
f2 = FollowUpItem("متى تبدأ مدة المسح؟")`;

// OpenUI adds a generic rule for libraries with charts and forms; this one has neither.
const OPENUI_DEFAULT_RULES =
  "## Important Rules\n- Choose components that best represent the content (tables for comparisons, charts for trends, forms for input, etc.)\n\n";

const prompt = library.prompt({ preamble, additionalRules: rules, examples: [example] });
if (!prompt.includes(OPENUI_DEFAULT_RULES)) throw new Error("OpenUI's default rules changed; update gen_prompt.tsx");

writeFileSync(
  new URL("../../backend/dalil_api/answer_components.md", import.meta.url),
  prompt.replace(OPENUI_DEFAULT_RULES, "") + "\n",
);
console.log("wrote web/backend/dalil_api/answer_components.md");
