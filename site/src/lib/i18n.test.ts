import { describe, expect, it } from "vitest";

import {
  languageIndependentPath,
  localeFromLanguages,
  preferredLocale,
  routerBasePath,
} from "@/lib/i18n";

describe("locale preference", () => {
  it("uses an explicit stored locale before browser languages", () => {
    expect(preferredLocale("en", ["zh-CN"])).toBe("en");
    expect(preferredLocale("zh-CN", ["en-US"])).toBe("zh-CN");
  });

  it("uses Chinese only for a Chinese primary browser language", () => {
    expect(localeFromLanguages(["zh-CN", "en-US"])).toBe("zh-CN");
    expect(localeFromLanguages(["zh-Hant-TW", "en-US"])).toBe("zh-CN");
    expect(localeFromLanguages(["en-US", "zh-Hant-TW"])).toBe("en");
    expect(localeFromLanguages(["fr", "zh-Hant-TW"])).toBe("en");
    expect(localeFromLanguages(["en-US", "ja"])).toBe("en");
    expect(localeFromLanguages(["fr-FR", "ja-JP"])).toBe("en");
    expect(localeFromLanguages([])).toBe("en");
    expect(preferredLocale("invalid", ["en-GB"])).toBe("en");
  });
});

describe("language-independent paths", () => {
  it("removes only a leading documentation locale", () => {
    expect(languageIndependentPath("/en/docs/quick-start")).toBe("/docs/quick-start");
    expect(languageIndependentPath("/zh-CN/docs/quick-start")).toBe("/docs/quick-start");
    expect(languageIndependentPath("/en")).toBe("/");
    expect(languageIndependentPath("/docs/en/examples")).toBe("/docs/en/examples");
  });
});

describe("router base path", () => {
  it("normalizes root and subdirectory deployments", () => {
    expect(routerBasePath("/")).toBe("/");
    expect(routerBasePath("/mizuoto/")).toBe("/mizuoto");
  });
});
