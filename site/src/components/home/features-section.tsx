import Link from "fumadocs-core/link";
import { ArrowUpRight, BookOpen, GraduationCap, Speech, Users, WandSparkles } from "lucide-react";

import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

/** One card per screen of the web UI, in the order of its navigation rail (with its icon). */
const FEATURE_ICONS = [Speech, BookOpen, WandSparkles, Users, GraduationCap] as const;

export default function FeaturesSection() {
  const copy = useSiteCopy().home.features;

  return (
    <PageSection id="features" labelledBy="features-title">
      <PageReveal className="page-section-heading is-centered">
        <span>FEATURES</span>
        <h2 id="features-title">{copy.title}</h2>
        <p>{copy.description}</p>
      </PageReveal>

      <div className="home-capability-grid page-spotlight">
        {copy.items.map(({ title, description, href }, index) => {
          const Icon = FEATURE_ICONS[index];
          return (
            <PageReveal
              className={`home-capability-cell${index === 0 ? " is-wide" : ""}`}
              index={index + 1}
              key={title}
            >
              <Link className="home-capability-card home-feature-card page-interactive" href={href}>
                <span className="home-capability-icon page-interactive-part" aria-hidden="true">
                  <Icon />
                </span>
                <h3>{title}</h3>
                <p>{description}</p>
                <span className="home-feature-link">
                  {copy.open}
                  <ArrowUpRight className="page-interactive-part" aria-hidden="true" />
                </span>
              </Link>
            </PageReveal>
          );
        })}
      </div>
    </PageSection>
  );
}
