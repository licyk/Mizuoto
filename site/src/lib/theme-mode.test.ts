import { describe, expect, it } from "vitest";

import {
  DEFAULT_THEME_MODE,
  isThemeMode,
  nextThemeMode,
  resolveThemeMode,
  THEME_MODES,
} from "@/lib/theme-mode";

describe("theme modes", () => {
  it("offers follow-system alongside light and dark", () => {
    expect(THEME_MODES).toEqual(["system", "light", "dark"]);
  });

  it("defaults to following the system", () => {
    expect(DEFAULT_THEME_MODE).toBe("system");
    expect(THEME_MODES[0]).toBe(DEFAULT_THEME_MODE);
  });

  it("recognises only the three modes", () => {
    for (const mode of THEME_MODES) expect(isThemeMode(mode)).toBe(true);
    for (const value of ["", "Light", "auto", null, undefined, 1]) {
      expect(isThemeMode(value)).toBe(false);
    }
  });
});

describe("resolveThemeMode", () => {
  it("keeps a stored mode", () => {
    expect(resolveThemeMode("dark")).toBe("dark");
    expect(resolveThemeMode("system")).toBe("system");
  });

  it("falls back to the default while storage is unread or invalid", () => {
    // `next-themes` reports `undefined` until it has read storage.
    expect(resolveThemeMode(undefined)).toBe(DEFAULT_THEME_MODE);
    expect(resolveThemeMode(null)).toBe(DEFAULT_THEME_MODE);
    expect(resolveThemeMode("sepia")).toBe(DEFAULT_THEME_MODE);
  });
});

describe("nextThemeMode", () => {
  it("cycles through every mode and returns to the start", () => {
    let mode = DEFAULT_THEME_MODE;
    const visited = [mode];

    for (let step = 0; step < THEME_MODES.length; step += 1) {
      mode = nextThemeMode(mode);
      visited.push(mode);
    }

    expect(visited).toEqual(["system", "light", "dark", "system"]);
  });

  it("reaches follow-system from every other mode", () => {
    // The stock two-icon switch could never return here; the cycle must.
    expect(nextThemeMode("dark")).toBe("system");
  });

  it("treats an unknown or unread value as the default", () => {
    expect(nextThemeMode(undefined)).toBe("light");
    expect(nextThemeMode("sepia")).toBe("light");
  });
});
