import { scrollToSection } from "@/lib/page-motion";
import { useSiteCopy } from "@/lib/site-copy";

/** Structural shape only, so any paged route can supply its own manifest. */
export interface RailSection {
  readonly id: string;
  readonly railLabel: string;
}

export interface SectionRailProps {
  sections: readonly RailSection[];
  activeId: string;
}

/**
 * Section rail.
 *
 * A pointer affordance that duplicates the header nav, so it is withdrawn on
 * touch and narrow viewports rather than reimplemented there.
 */
export default function SectionRail({ sections, activeId }: SectionRailProps) {
  const copy = useSiteCopy();

  return (
    <nav aria-label={copy.pageRailLabel} className="page-rail">
      <ul>
        {sections.map((section) => (
          <li key={section.id}>
            <button
              aria-current={activeId === section.id ? "true" : undefined}
              aria-label={section.railLabel}
              className="page-rail-dot"
              onClick={() => scrollToSection(section.id)}
              title={section.railLabel}
              type="button"
            >
              <span className="page-rail-label" aria-hidden="true">
                {section.railLabel}
              </span>
              <span className="page-rail-mark" aria-hidden="true" />
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
