import { useEffect, useState, type RefObject } from "react";

const REVEAL_SELECTOR = "[data-page-reveal]";
const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

/**
 * Section rail activation band. A section counts as active while it covers the
 * middle 10% of the viewport, which tracks the reading position far more
 * closely than a plain visibility threshold does.
 */
const ACTIVE_SECTION_MARGIN = "-45% 0px -45% 0px";

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia(REDUCED_MOTION_QUERY).matches;
}

/**
 * True when the browser can scrub animations against scroll position. When it
 * can, reveals are pure CSS (see `motion.css`) and the JavaScript fallback
 * must stay out of the way so the two never run at once.
 */
export function supportsScrollTimeline(): boolean {
  if (typeof CSS === "undefined" || typeof CSS.supports !== "function") return false;
  return CSS.supports("animation-timeline: view()");
}

/**
 * Fallback reveal for browsers without scroll-driven animations.
 *
 * The pending class is applied to `.page-reveal` wrappers only. Those wrappers
 * never carry hover styles, which is what keeps the reveal's `transition-delay`
 * from leaking into interaction timing.
 */
export function useScrollReveal(rootRef: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (supportsScrollTimeline()) return;

    const nodes = Array.from(root.querySelectorAll<HTMLElement>(REVEAL_SELECTOR));
    const reducedMotion = window.matchMedia(REDUCED_MOTION_QUERY);
    let observer: IntersectionObserver | null = null;

    function revealAll() {
      observer?.disconnect();
      observer = null;
      for (const node of nodes) node.classList.remove("is-reveal-pending");
    }

    if (reducedMotion.matches || typeof window.IntersectionObserver !== "function") {
      return;
    }

    const visibleBoundary = window.innerHeight * 0.9;
    const pendingNodes = nodes.filter((node) => {
      if (node.getBoundingClientRect().top <= visibleBoundary) return false;
      node.classList.add("is-reveal-pending");
      return true;
    });

    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.remove("is-reveal-pending");
          observer?.unobserve(entry.target);
        }
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.15 },
    );

    for (const node of pendingNodes) observer.observe(node);

    function handleReducedMotion(event: MediaQueryListEvent) {
      if (event.matches) revealAll();
    }

    reducedMotion.addEventListener("change", handleReducedMotion);

    return () => {
      observer?.disconnect();
      reducedMotion.removeEventListener("change", handleReducedMotion);
      for (const node of nodes) node.classList.remove("is-reveal-pending");
    };
  }, [rootRef]);
}

/**
 * Resolves the section currently occupying the middle of the viewport.
 *
 * Returns the first id in document order among the sections intersecting the
 * activation band, so a section taller than the band never hands activation to
 * its neighbour.
 */
export function useActiveSection(ids: readonly string[]): string {
  const [activeId, setActiveId] = useState(ids[0] ?? "");

  useEffect(() => {
    if (typeof window.IntersectionObserver !== "function") return;

    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => element !== null);

    if (elements.length === 0) return;

    const visible = new Set<string>();

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id);
          else visible.delete(entry.target.id);
        }

        const next = ids.find((id) => visible.has(id));
        if (next) setActiveId(next);
      },
      { rootMargin: ACTIVE_SECTION_MARGIN, threshold: 0 },
    );

    for (const element of elements) observer.observe(element);
    return () => observer.disconnect();
  }, [ids]);

  return activeId;
}

/** Scrolls a section into view, honouring the reduced-motion preference. */
export function scrollToSection(id: string): void {
  const element = document.getElementById(id);
  if (!element) return;

  element.scrollIntoView({
    behavior: prefersReducedMotion() ? "auto" : "smooth",
    block: "start",
  });
}
