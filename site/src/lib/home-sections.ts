import { DEFAULT_LOCALE, type Locale } from "@/lib/i18n";
import { getSiteCopy } from "@/lib/site-copy";

/**
 * The homepage section manifest.
 *
 * Single source of truth for section identity: the route renders from it, the
 * section rail navigates by it, and the hero's "next section" button reads the
 * entry that follows the hero, so an id can never drift between them.
 */
export interface HomeSectionMeta {
  /** DOM id and anchor target. */
  readonly id: string;
  /** Accessible name used by the section rail. */
  readonly railLabel: string;
}

const HOME_SECTION_KEYS = ["hero", "features", "workflow", "highlights", "install"] as const;

export function homeSections(locale: Locale): readonly HomeSectionMeta[] {
  const labels = getSiteCopy(locale).home.sections;
  return HOME_SECTION_KEYS.map((id, index) => ({ id, railLabel: labels[index] }));
}

export const HOME_SECTIONS = homeSections(DEFAULT_LOCALE);

export const HOME_SECTION_IDS: readonly string[] = HOME_SECTIONS.map((section) => section.id);

/**
 * The section immediately after `id`, or `undefined` at the end.
 *
 * The hero's scroll button uses this rather than naming a target directly, so
 * inserting a section can never leave it skipping past the new one.
 */
export function nextHomeSection(
  id: string,
  locale: Locale = DEFAULT_LOCALE,
): HomeSectionMeta | undefined {
  const sections = homeSections(locale);
  const index = sections.findIndex((section) => section.id === id);
  return index < 0 ? undefined : sections[index + 1];
}
