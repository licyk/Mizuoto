import type { ReactNode } from "react";

export interface HomeSectionProps {
  /** Extra class on the full-bleed decoration layer; omit for no decoration. */
  backdropClassName?: string;
  /** DOM id; must match an entry in the section manifest. */
  id: string;
  /** Id of the heading that names this section. */
  labelledBy?: string;
  tone?: "default" | "band" | "ink";
  /**
   * Sections fill the viewport by default. `fill={false}` opts out for the
   * outro, which has nothing to gain from the extra height.
   */
  fill?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * Full-viewport section primitive.
 *
 * The section is the snap area and the centring grid; `.page-section-body` is
 * the single grid item and the carrier for the scroll-pass animation, so
 * transforms never disturb the section's own box or its snap position.
 */
export default function PageSection({
  id,
  labelledBy,
  tone = "default",
  fill = true,
  className = "",
  backdropClassName,
  children,
}: HomeSectionProps) {
  const classNames = ["page-section", `is-${tone}`, fill ? "" : "is-auto", className]
    .filter(Boolean)
    .join(" ");

  return (
    <section aria-labelledby={labelledBy} className={classNames} id={id}>
      {backdropClassName ? (
        <div className={`page-section-backdrop ${backdropClassName}`} aria-hidden="true" />
      ) : null}
      <div className="page-section-body">
        <div className="page-shell">{children}</div>
      </div>
    </section>
  );
}
