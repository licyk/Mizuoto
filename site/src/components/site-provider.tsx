import { i18nProvider } from "fumadocs-ui/i18n";
import { RootProvider } from "fumadocs-ui/provider/tanstack";
import { useEffect, useState, type ReactNode } from "react";

import SearchDialog from "@/components/search";
import {
  DEFAULT_LOCALE,
  isLocale,
  preferredLocale,
  readStoredLocale,
  storeLocale,
} from "@/lib/i18n";
import { translations } from "@/lib/layout.shared";

export function SiteProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState(DEFAULT_LOCALE);
  const provider = i18nProvider(translations, locale);

  useEffect(() => {
    setLocale(preferredLocale(readStoredLocale(), navigator.languages));
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  function changeLocale(nextLocale: string) {
    if (!isLocale(nextLocale)) return;

    storeLocale(nextLocale);
    setLocale(nextLocale);
  }

  return (
    <RootProvider
      i18n={{ ...provider, locale, onLocaleChange: changeLocale }}
      search={{ SearchDialog }}
    >
      {children}
    </RootProvider>
  );
}
