import type { CSSProperties, ReactNode } from "react";

export interface PageRevealProps {
  /** Stagger position within a group; drives the reveal offset. */
  index?: number;
  element?: "div" | "li";
  className?: string;
  children: ReactNode;
}

/**
 * Reveal wrapper — the enter channel.
 *
 * A `.page-reveal` element must never carry hover styles. Keeping the two
 * channels on separate nodes is what stops the stagger offset from leaking
 * into interaction timing, which is exactly how the previous homepage ended up
 * with hover states that lagged on entry and cut on exit.
 */
export default function PageReveal({
  index = 0,
  element: Element = "div",
  className = "",
  children,
}: PageRevealProps) {
  return (
    <Element
      className={`page-reveal ${className}`.trim()}
      data-page-reveal=""
      style={{ "--page-stagger": index } as CSSProperties}
    >
      {children}
    </Element>
  );
}
