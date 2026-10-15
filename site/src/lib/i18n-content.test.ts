import { readdirSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const docsDirectory = new URL("../../content/docs/", import.meta.url);

function documentSlugs(pages: string[]): string[] {
  return pages.filter((page) => !page.startsWith("---"));
}

describe("localized documentation", () => {
  it("has an English document for every Chinese source document", () => {
    const names = readdirSync(docsDirectory);
    const chineseDocuments = names.filter(
      (name) => name.endsWith(".mdx") && !name.endsWith(".en.mdx"),
    );

    for (const chineseDocument of chineseDocuments) {
      const englishDocument = chineseDocument.replace(/\.mdx$/, ".en.mdx");
      expect(names, `${englishDocument} is missing`).toContain(englishDocument);
    }
  });

  it("keeps the same document slugs and order in both sidebars", () => {
    const chinese = JSON.parse(readFileSync(new URL("meta.json", docsDirectory), "utf8")) as {
      pages: string[];
    };
    const english = JSON.parse(readFileSync(new URL("meta.en.json", docsDirectory), "utf8")) as {
      pages: string[];
    };

    expect(documentSlugs(english.pages)).toEqual(documentSlugs(chinese.pages));
  });
});
