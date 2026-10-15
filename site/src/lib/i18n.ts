import { defineI18n } from "fumadocs-core/i18n";

export const LOCALES = ["zh-CN", "en"] as const;
export type Locale = (typeof LOCALES)[number];

export const DEFAULT_LOCALE: Locale = "zh-CN";
export const LOCALE_STORAGE_KEY = "mizuoto.site.locale";

export const i18n = defineI18n({
  defaultLanguage: DEFAULT_LOCALE,
  fallbackLanguage: null,
  hideLocale: "default-locale",
  languages: [...LOCALES],
});

export function languageIndependentPath(path: string): string {
  return path.replace(/^\/(?:zh-CN|en)(?=\/|$)/, "") || "/";
}

export function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && LOCALES.includes(value as Locale);
}

export function localeFromLanguages(languages: readonly string[]): Locale {
  const primaryLanguage = languages[0]?.toLowerCase();
  return primaryLanguage?.startsWith("zh") ? "zh-CN" : "en";
}

export function preferredLocale(
  storedLocale: string | null | undefined,
  languages: readonly string[],
): Locale {
  return isLocale(storedLocale) ? storedLocale : localeFromLanguages(languages);
}

export function readStoredLocale(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(LOCALE_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function storeLocale(locale: Locale): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
  } catch {
    // A blocked storage API must not prevent language selection.
  }
}

export function routerBasePath(baseUrl: string): string {
  const normalized = `/${baseUrl}`.replace(/\/{2,}/g, "/").replace(/\/$/, "");
  return normalized || "/";
}
