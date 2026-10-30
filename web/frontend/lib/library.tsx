"use client";

/**
 * The OpenUI component library the answer model writes with.
 *
 * Daleel's own components refer to passages by number only. Quote takes the
 * model's copy of the words but shows the source's own span, found by
 * findVerbatim; a quote that is not in the passage is shown as unverified.
 * OpenUI's own built-ins are swapped for same-signature versions where they
 * could not show citations or hardcode English.
 */
import { createLibrary, defineComponent, useIsStreaming, useTriggerAction } from "@openuidev/react-lang";
import {
  FollowUpItem as UIFollowUpItem, Steps as UISteps, StepsItem as UIStepsItem, Tag, TextCallout,
} from "@openuidev/react-ui";
import { z } from "zod/v4";
import { CiteChip, CiteChips, CitedText } from "@/components/CitedText";
import { findVerbatim } from "./arabic";
import { usePassage } from "./passages";

const Lead = defineComponent({
  name: "Lead",
  description: "The direct answer in one to three sentences, with [n] citations. Always first.",
  props: z.object({
    text: z.string().describe("Markdown with [n] citations after each claim"),
  }),
  component: ({ props }) => <CitedText text={props.text ?? ""} className="ans-lead" />,
});

const Prose = defineComponent({
  name: "Prose",
  description: "A paragraph of explanation with [n] citations, optionally under a short heading.",
  props: z.object({
    text: z.string().describe("Markdown with [n] citations; may use '- ' bullet lines"),
    heading: z.string().optional().describe("A short heading, two to five words"),
  }),
  component: ({ props }) => (
    <section className="ans-prose">
      {props.heading ? <h3 className="ans-h">{props.heading}</h3> : null}
      <CitedText text={props.text ?? ""} />
    </section>
  ),
});

function QuoteCard({ source, words }: { source: number; words: string }) {
  const streaming = useIsStreaming();
  const p = usePassage(source);
  const haystack = p ? p.text : "";
  const span = words ? findVerbatim(haystack, words) : null;
  const verified = Boolean(p && span);
  const pending = streaming && !verified;
  const shown = verified && span ? haystack.slice(span[0], span[1]) : words;
  const hadith = p?.refs?.hadith;
  return (
    <figure className={`ans-quote ${verified ? "is-verified" : pending ? "is-pending" : "is-unverified"}`}>
      <blockquote dir="rtl">{shown}</blockquote>
      <figcaption>
        <CiteChip n={source} />
        <span className="ans-quote-cite">{p?.citation ?? "مصدر غير موجود"}</span>
        {hadith?.hadith_number ? <span className="ans-quote-meta">رقم {hadith.hadith_number}</span> : null}
        {p?.grade ? <Tag size="sm" variant="info" text={`الألباني: ${p.grade}`} /> : null}
        {verified ? (
          <Tag size="sm" variant="success" text="مطابق للمصدر" />
        ) : pending ? null : (
          <Tag size="sm" variant="warning" text="لم يُعثر عليه حرفيًا في المصدر" />
        )}
      </figcaption>
    </figure>
  );
}

const Quote = defineComponent({
  name: "Quote",
  description:
    "A verbatim quotation from one passage: the decisive words of a ruling, hadith or ayah. The words are checked against the passage and shown from it.",
  props: z.object({
    source: z.number().describe("The passage number the words are copied from"),
    words: z.string().describe("Contiguous words copied exactly from that passage, 5 to 40 words"),
  }),
  component: ({ props }) => <QuoteCard source={props.source} words={props.words ?? ""} />,
});

const View = defineComponent({
  name: "View",
  description: "One position within Views",
  props: z.object({
    holder: z.string().describe("Who holds it: a madhhab, scholar or group, as the passages name them"),
    position: z.string().describe("The position and its evidence in one to three sentences, with [n] citations"),
    // Optional: the position text already carries [n]. When it was required, a View
    // the model wrote without it failed validation and the whole position vanished.
    sources: z.array(z.number()).optional().describe("Passage numbers that state this position"),
  }),
  component: () => null,
});

const Views = defineComponent({
  name: "Views",
  description: "Side-by-side positions when the passages show more than one view on the question.",
  props: z.object({
    title: z.string().describe("What the positions differ on, as a short phrase"),
    items: z.array(View.ref),
  }),
  component: ({ props }) => {
    const items = props.items ?? [];
    if (!items.length) return null;
    return (
      <section className="ans-views">
        <h3 className="ans-h">{props.title}</h3>
        <div className="ans-views-grid" data-count={items.length}>
          {items.map((item, i) => (
            <div key={i} className="ans-view">
              <div className="ans-view-holder">{item?.props?.holder}</div>
              <CitedText text={item?.props?.position ?? ""} />
              {item?.props?.sources?.length ? (
                <div className="ans-view-sources">
                  <CiteChips nums={item.props.sources} />
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </section>
    );
  },
});

const Step = defineComponent({
  name: "Step",
  description: "One step within Steps",
  props: z.object({
    title: z.string(),
    details: z.string().describe("Markdown with [n] citations"),
  }),
  component: () => null,
});

const Steps = defineComponent({
  name: "Steps",
  description: "An ordered procedure, only when a passage itself describes one step by step.",
  props: z.object({ items: z.array(Step.ref) }),
  component: ({ props }) => (
    <section className="ans-steps">
      <UISteps>
        {(props.items ?? []).map((item, i) => (
          <UIStepsItem
            key={i}
            number={i + 1}
            title={item?.props?.title ?? ""}
            details={<CitedText text={item?.props?.details ?? ""} />}
          />
        ))}
      </UISteps>
    </section>
  ),
});

const Gap = defineComponent({
  name: "Gap",
  description: "What the passages do not answer. Use whenever they answer only part of the question, or none of it.",
  props: z.object({ text: z.string() }),
  component: ({ props }) => (
    <TextCallout
      className="ans-callout" variant="neutral" title="ما لا تغطيه المصادر"
      description={<CitedText text={props.text ?? ""} />}
    />
  ),
});

const Referral = defineComponent({
  name: "Referral",
  description: "For a question about the asker's own situation: tell them to take their case to a qualified scholar.",
  props: z.object({ text: z.string() }),
  component: ({ props }) => (
    <TextCallout
      className="ans-callout" variant="info" title="استشر أهل العلم"
      description={<CitedText text={props.text ?? ""} />}
    />
  ),
});

// OpenUI's Table renders cells as plain text, so [n] would show raw. Same
// signature as the built-in (Table(columns: Col[]), column-oriented), cited cells.
const Col = defineComponent({
  name: "Col",
  description: "Column definition — holds label + data array",
  props: z.object({
    label: z.string(),
    data: z.array(z.string()).describe("One cell per row; may carry [n] citations"),
  }),
  component: () => null,
});

const Table = defineComponent({
  name: "Table",
  description: "Data table — column-oriented. Each Col holds its own data array.",
  props: z.object({ columns: z.array(Col.ref) }),
  component: ({ props }) => {
    const cols = props.columns ?? [];
    const rows = Math.max(0, ...cols.map((c) => c?.props?.data?.length ?? 0));
    return (
      <div className="ans-table-wrap">
        <table className="ans-table">
          <thead>
            <tr>{cols.map((c, i) => <th key={i}>{c?.props?.label}</th>)}</tr>
          </thead>
          <tbody>
            {Array.from({ length: rows }, (_, r) => (
              <tr key={r}>
                {cols.map((c, i) => (
                  <td key={i}><CitedText text={String(c?.props?.data?.[r] ?? "")} /></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  },
});

// OpenUI's FollowUpBlock hardcodes an English "Related Queries" header, so this
// keeps its name and props (the prompt is unchanged) and its item primitive.
const FollowUpItem = defineComponent({
  name: "FollowUpItem",
  description: "Clickable follow-up suggestion — when clicked, sends text as user message",
  props: z.object({ text: z.string() }),
  component: () => null,
});

const FollowUpBlock = defineComponent({
  name: "FollowUpBlock",
  description: "List of clickable follow-up suggestions placed at the end of a response",
  props: z.object({ items: z.array(FollowUpItem.ref) }),
  component: function FollowUps({ props }) {
    const trigger = useTriggerAction();
    const items = props.items ?? [];
    return (
      <nav className="ans-followups" aria-label="أسئلة ذات صلة">
        <h3 className="ans-h">أسئلة ذات صلة</h3>
        {items.map((item, i) => {
          const text = String(item?.props?.text ?? "");
          return <UIFollowUpItem key={i} text={text} dir="auto" onClick={() => trigger(text)} />;
        })}
      </nav>
    );
  },
});

const Answer = defineComponent({
  name: "Answer",
  description: "The whole answer: blocks stacked top to bottom.",
  props: z.object({
    blocks: z.array(
      z.union([
        Lead.ref, Prose.ref, Quote.ref, Views.ref, Steps.ref, Gap.ref, Referral.ref,
        Table.ref, FollowUpBlock.ref,
      ]),
    ),
  }),
  component: ({ props, renderNode }) => (
    <article className="ans">
      {(props.blocks ?? []).map((b, i) => (
        <div key={i} className="ans-block">
          {renderNode(b)}
        </div>
      ))}
    </article>
  ),
});

export const daleelLibrary = createLibrary({
  root: "Answer",
  components: [
    Answer, Lead, Prose, Quote, Views, View, Steps, Step, Gap, Referral,
    Table, Col, FollowUpBlock, FollowUpItem,
  ],
});
