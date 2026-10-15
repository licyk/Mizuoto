import { zhCN } from "@fumadocs/language/zh-cn";
import { uiTranslations } from "fumadocs-ui/i18n";
import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

import ThemeToggle from "@/components/theme-toggle";
import { DEFAULT_LOCALE, i18n, type Locale } from "@/lib/i18n";

export const REPOSITORY_URL = "https://github.com/licyk/Mizuoto";

export const translations = i18n
  .translations()
  .extend(uiTranslations())
  .preset("zh-CN", zhCN())
  .add({
    "zh-CN": { displayName: "简体中文" },
    en: { displayName: "English" },
  });

export function baseOptions(locale: Locale = DEFAULT_LOCALE): BaseLayoutProps {
  return {
    // One cycling button in place of the stock two-icon switch, everywhere the
    // layout renders a theme control.
    slots: { themeSwitch: ThemeToggle },
    nav: {
      title: "Mizuoto",
      url: "/",
      // The hero owns the top of the page; the header earns its backdrop on scroll.
      transparentMode: "top",
    },
    githubUrl: REPOSITORY_URL,
    links: [
      {
        text: locale === "zh-CN" ? "文档" : "Docs",
        url: "/docs",
        active: "nested-url",
      },
    ],
  };
}

export function docsOptions(locale: Locale = DEFAULT_LOCALE): BaseLayoutProps {
  return {
    slots: { themeSwitch: ThemeToggle },
    nav: {
      title: "Mizuoto",
      url: "/",
    },
    githubUrl: REPOSITORY_URL,
  };
}
