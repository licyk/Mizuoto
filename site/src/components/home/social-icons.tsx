import type { SVGProps } from "react";

/*
 * Brand marks lucide does not ship.
 *
 * Drawn on lucide's own grid — 24x24, `currentColor` strokes at width 2 with
 * round joins — so they sit beside `Rss` in the footer without reading as a
 * different icon set.
 */
function brandIconProps(props: SVGProps<SVGSVGElement>): SVGProps<SVGSVGElement> {
  return {
    xmlns: "http://www.w3.org/2000/svg",
    width: 24,
    height: 24,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    ...props,
  };
}

export function GithubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...brandIconProps(props)}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}

/* The bilibili television: two antennae over a rounded set with two eyes. */
export function BilibiliIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg {...brandIconProps(props)}>
      <path d="m7 3 3 3" />
      <path d="m17 3-3 3" />
      <rect width="18" height="14" x="3" y="6" rx="4" />
      <path d="M8.5 12v1.5" />
      <path d="M15.5 12v1.5" />
    </svg>
  );
}
