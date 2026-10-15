import { Boxes, Layers, Network, Radio, Server } from "lucide-react";

import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

const HIGHLIGHT_ICONS = [Layers, Boxes, Server, Radio, Network] as const;

export default function HighlightsSection() {
  const copy = useSiteCopy().home.highlights;

  return (
    <PageSection id="highlights" labelledBy="highlights-title">
      <PageReveal className="page-section-heading">
        <span>WHY MIZUOTO</span>
        <h2 id="highlights-title">{copy.title}</h2>
        <p>{copy.description}</p>
      </PageReveal>

      <ul className="home-highlight-list page-spotlight">
        {copy.items.map(({ title, description }, index) => {
          const Icon = HIGHLIGHT_ICONS[index];
          return (
            <PageReveal element="li" index={index + 1} key={title}>
              <div className="home-highlight page-interactive">
                <span className="home-highlight-icon page-interactive-part" aria-hidden="true">
                  <Icon />
                </span>
                <div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </div>
              </div>
            </PageReveal>
          );
        })}
      </ul>
    </PageSection>
  );
}
