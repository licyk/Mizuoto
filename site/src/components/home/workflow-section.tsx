import { GraduationCap, Library, PackageOpen, Speech } from "lucide-react";

import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

const WORKFLOW_ICONS = [PackageOpen, Library, Speech, GraduationCap] as const;

export default function WorkflowSection() {
  const copy = useSiteCopy().home.workflow;

  return (
    <PageSection id="workflow" labelledBy="workflow-title" tone="band">
      <PageReveal className="page-section-heading">
        <span>WORKFLOW</span>
        <h2 id="workflow-title">{copy.title}</h2>
        <p>{copy.description}</p>
      </PageReveal>

      <ol className="home-workflow-grid page-spotlight">
        {copy.steps.map(({ title, description }, index) => {
          const Icon = WORKFLOW_ICONS[index];
          const label = String(index + 1).padStart(2, "0");
          return (
            <PageReveal element="li" index={index + 1} key={label}>
              {/* The step body carries the interaction channel; the `<li>` is the
                reveal wrapper and must stay out of it. */}
              <div className="home-workflow-step page-interactive">
                <div className="home-workflow-number">{label}</div>
                <span className="home-workflow-icon page-interactive-part" aria-hidden="true">
                  <Icon />
                </span>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
            </PageReveal>
          );
        })}
      </ol>
    </PageSection>
  );
}
