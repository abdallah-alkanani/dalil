/**
 * Writes web/backend/dalil_api/answer_components.md from the component library:
 * the OpenUI instructions the API appends to prompts/answering_system_prompt.md.
 *
 *     npm run prompt
 */
import { writeFileSync } from "node:fs";
import { library } from "../lib/library";

const preamble = `# Writing the answer on the website

On the website the search has already run: the numbered passages in the user message are its results, in rank order. Every rule above still applies. The answer is written as openui-lang with the components below, and the interface shows the question, so start with the short answer.`;

const rules = [
  "root is always Answer([...]). Map the answer shape above onto it: الجواب المختصر is the Lead; the evidence and the views follow as Quote, Prose and Views blocks, Quran first, then hadith, then the scholars' words.",
  "Refer to a passage by its number. Put [n] after every statement that rests on passage n, and name who said it and in which book in the sentence itself.",
  "Quote pastes text from one passage: whole sentences, exactly as in the passage, with the part that answers the question in **bold**. The interface checks the words against the passage and shows the passage's own text.",
  "Views holds the views when the passages contain more than one, each with who holds it as the passages name them.",
  "Your own connection of the evidence goes in a Prose whose heading is «استنتاج من النصوص أعلاه، وليس نصًا لأحد العلماء», with [n] for the passages it rests on.",
  "Gap says what the passages do not answer; when nothing answers the question, the answer is a Lead saying «لم أجد نصًا يجيب عن السؤال في قاعدة البيانات» and nothing else.",
  "Referral tells the asker to take a personal case to a qualified scholar.",
  "Steps only when a passage sets out a procedure; Table only to set short facts from several passages side by side.",
  "End with a FollowUpBlock of two or three follow-up questions in the user's language that the passages could answer.",
];

const example = `root = Answer([lead, q1, views, q2, fu])
lead = Lead("قال ابن عثيمين في فقه العبادات إن المسح على الخفين ثابت بالسنة المتواترة [2]، وذكر من شروطه أن يلبسهما على طهارة [2].")
q1 = Quote(2, "الجواب: **المسح على الخفين مما تواترت به السنة عن النبي صلى الله عليه وسلم**")
views = Views("ستر محل الفرض", [v1, v2])
v1 = View("ابن قدامة في الكافي", "يشترط أن يستر الخف محل الفرض [7].", [7])
v2 = View("ابن تيمية في الفتاوى الكبرى", "يجوز المسح على الخف ما دام اسمه باقيًا [5].", [5])
q2 = Quote(1, "جعل النبي صلى الله عليه وسلم **للمقيم يوماً وليلة، وللمسافر ثلاثة أيام ولياليهن**")
fu = FollowUpBlock([f1, f2])
f1 = FollowUpItem("متى تبدأ مدة المسح؟")
f2 = FollowUpItem("ما حكم المسح على الجوربين؟")`;

writeFileSync(
  new URL("../../backend/dalil_api/answer_components.md", import.meta.url),
  library.prompt({ preamble, additionalRules: rules, examples: [example] }) + "\n",
);
console.log("wrote web/backend/dalil_api/answer_components.md");
