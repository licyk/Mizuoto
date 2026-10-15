/**
 * The three theme modes the site cycles through.
 *
 * `system` comes first because it is the default: `RootProvider` hands
 * `next-themes` a `defaultTheme` of `system`, so an untouched visit follows the
 * operating system. Keeping it in the cycle is what lets a visitor get back to
 * that state after choosing light or dark.
 */
export const THEME_MODES = ["system", "light", "dark"] as const;

export type ThemeMode = (typeof THEME_MODES)[number];

export const DEFAULT_THEME_MODE: ThemeMode = "system";

export function isThemeMode(value: unknown): value is ThemeMode {
  return typeof value === "string" && (THEME_MODES as readonly string[]).includes(value);
}

/**
 * The stored theme, or the default when it is missing or unrecognised.
 *
 * `next-themes` reports `undefined` until it has read storage, so the button
 * renders the default rather than flickering through a wrong icon.
 */
export function resolveThemeMode(value: string | null | undefined): ThemeMode {
  return isThemeMode(value) ? value : DEFAULT_THEME_MODE;
}

/** The next mode in the cycle, wrapping back to the start. */
export function nextThemeMode(current: string | null | undefined): ThemeMode {
  const index = THEME_MODES.indexOf(resolveThemeMode(current));
  return THEME_MODES[(index + 1) % THEME_MODES.length];
}
