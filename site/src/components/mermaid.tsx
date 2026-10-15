import { useTheme } from "fumadocs-ui/provider/base";
import { useEffect, useId, useState } from "react";

/** Mermaid's ES module build, pinned to a major version. */
const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@12/dist/mermaid.esm.min.mjs";

interface MermaidApi {
  initialize(config: Record<string, unknown>): void;
  render(id: string, text: string): Promise<{ svg: string }>;
}

/**
 * A Mermaid diagram from the development docs (``<Mermaid chart="…" />``).
 *
 * Rendered in the browser only: the static build keeps the source as a code
 * block. Mermaid is loaded from the CDN on demand rather than bundled: pages
 * without a diagram never download it, and the build stays out of its sources
 * (Tailwind found stray class candidates in them in the server build only, which
 * gave the server and the client different stylesheets). It follows the site's
 * light/dark theme.
 */
export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, "");
  const { resolvedTheme } = useTheme();
  const [svg, setSvg] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (import(/* @vite-ignore */ MERMAID_URL) as Promise<{ default: MermaidApi }>).then(async ({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: resolvedTheme === "dark" ? "dark" : "default",
        fontFamily: "inherit",
      });
      const { svg: rendered } = await mermaid.render(`mermaid-${id}`, chart);
      if (!cancelled) setSvg(rendered);
    });
    return () => {
      cancelled = true;
    };
  }, [chart, id, resolvedTheme]);

  if (svg === null) {
    return (
      <pre className="overflow-x-auto rounded-lg border p-4 text-sm">
        <code>{chart}</code>
      </pre>
    );
  }
  // eslint-disable-next-line react/no-danger -- mermaid's own SVG, rendered with securityLevel "strict".
  return <div className="my-6 flex justify-center overflow-x-auto" dangerouslySetInnerHTML={{ __html: svg }} />;
}
