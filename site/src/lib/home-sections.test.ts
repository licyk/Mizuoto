import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { HOME_SECTION_IDS, HOME_SECTIONS, nextHomeSection } from "@/lib/home-sections";
import { DEFAULT_LOCALE, type Locale } from "@/lib/i18n";
import { baseOptions } from "@/lib/layout.shared";
import { getSiteCopy } from "@/lib/site-copy";

describe("home section manifest", () => {
  it("keeps section ids unique", () => {
    expect(new Set(HOME_SECTION_IDS).size).toBe(HOME_SECTION_IDS.length);
  });

  it("starts at the hero", () => {
    expect(HOME_SECTION_IDS[0]).toBe("hero");
  });

  it("gives every section a rail label", () => {
    for (const section of HOME_SECTIONS) {
      expect(section.railLabel.length).toBeGreaterThan(0);
    }
  });
});

describe("nextHomeSection", () => {
  it("advances exactly one section", () => {
    expect(nextHomeSection("hero")?.id).toBe(HOME_SECTION_IDS[1]);
    expect(nextHomeSection(HOME_SECTION_IDS[2])?.id).toBe(HOME_SECTION_IDS[3]);
  });

  it("does not skip ahead to the highlights section", () => {
    // The hero button stops at whatever section immediately follows the hero.
    expect(nextHomeSection("hero")?.id).toBe("features");
  });

  it("returns undefined past the last section and for unknown ids", () => {
    expect(nextHomeSection(HOME_SECTION_IDS.at(-1) as string)).toBeUndefined();
    expect(nextHomeSection("not-a-section")).toBeUndefined();
  });
});

describe("header navigation", () => {
  const links = baseOptions().links ?? [];

  it("carries no in-page section jumps", () => {
    const anchors = links.filter(
      (link) => "url" in link && typeof link.url === "string" && link.url.startsWith("/#"),
    );

    expect(anchors).toEqual([]);
  });

  it("carries no custom action button", () => {
    expect(links.filter((link) => link.type === "custom")).toEqual([]);
  });

  it("links the documentation and the repository", () => {
    expect(links).toEqual([{ text: "文档", url: "/docs", active: "nested-url" }]);
    expect(baseOptions().githubUrl).toBe("https://github.com/licyk/Mizuoto");
  });
});

describe("home copy", () => {
  const meta = JSON.parse(
    readFileSync(new URL("../../content/docs/meta.json", import.meta.url), "utf8"),
  ) as { pages: string[] };

  it.each(["zh-CN", "en"] satisfies Locale[])("links every %s feature to a listed page", (locale) => {
    for (const { href } of getSiteCopy(locale).home.features.items) {
      expect(meta.pages).toContain(href.replace(/^\/docs\//, ""));
    }
  });

  it("gives both locales the same number of entries", () => {
    const zh = getSiteCopy(DEFAULT_LOCALE).home;
    const en = getSiteCopy("en").home;
    expect(en.sections.length).toBe(zh.sections.length);
    expect(en.features.items.length).toBe(zh.features.items.length);
    expect(en.workflow.steps.length).toBe(zh.workflow.steps.length);
    expect(en.highlights.items.length).toBe(zh.highlights.items.length);
  });
});
