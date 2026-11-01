import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "عن دليل",
  description:
    "دليل: بحث عربي في ٥٤٢ كتابًا من القرآن والسنة وأقوال أهل العلم، وكل ما يبحث فيه متاح للتنزيل.",
};

const LINKS = [
  {
    label: "GitHub",
    title: "الكود المصدري",
    url: "abdallah-alkanani/dalil",
    href: "https://github.com/abdallah-alkanani/dalil",
  },
  {
    label: "Hugging Face",
    title: "الكتب وقواعد البحث والمتجهات",
    url: "datasets/Alkanani/dalil",
    href: "https://huggingface.co/datasets/Alkanani/dalil",
  },
];

export default function AboutPage() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link href="/" className="wordmark" aria-label="دليل — البداية">
          دليل
        </Link>
        <span className="topbar-sub">من القرآن والسنة وأقوال أهل العلم</span>
        <span className="topbar-spacer" />
        <Link href="/" className="topbar-about">
          ابدأ البحث
        </Link>
      </header>

      <main className="about">
        <p className="about-kicker">عن المشروع</p>
        <h1 className="about-title">دليل</h1>
        <p className="about-lead">
          محرّك بحث عربي في ٥٤٢ كتابًا من القرآن والحديث والتفسير والفقه والعقيدة والسيرة. تعيد
          الإجابة النصوص نفسها، ويأتي مع كل نص اسم العالم والكتاب والموضع.
        </p>
        <p className="about-body">
          وما يبحث فيه دليل كلّه متاح للتنزيل: الكتب في صورتها الأصلية والمنقّحة، وقواعد البحث
          بالكلمات والمتجهات، و٩٢٧٬٧٤٤ متجهة محسوبة سلفًا في فهرس FAISS دقيق. لا يُحسب عند
          البحث إلا متجهة السؤال.
        </p>

        <div className="about-links">
          {LINKS.map((link) => (
            <a
              key={link.label}
              className="about-link"
              href={link.href}
              target="_blank"
              rel="noreferrer"
            >
              <span className="about-link-label">{link.label}</span>
              <span className="about-link-title">{link.title}</span>
              <span className="about-link-url" dir="ltr">
                {link.url}
              </span>
            </a>
          ))}
        </div>

        <p className="about-note">
          الموقع مفتوح للجميع، بلا تسجيل. تُحفظ الأسئلة وأسماء الكتب التي بُحث فيها لتحسين البحث،
          ولا يُحفظ اسم من يسأل إلا إذا كتبه.
        </p>

        <Link className="about-start" href="/">
          اسأل سؤالك
        </Link>
      </main>
    </div>
  );
}
