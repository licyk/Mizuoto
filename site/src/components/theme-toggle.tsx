import { useTheme } from "fumadocs-ui/provider/base";
import { Monitor, Moon, Sun, type LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { flushSync } from "react-dom";

import { useSiteCopy } from "@/lib/site-copy";
import {
  DEFAULT_THEME_MODE,
  nextThemeMode,
  resolveThemeMode,
  type ThemeMode,
} from "@/lib/theme-mode";

const MODE_ICONS: Record<ThemeMode, LucideIcon> = {
  system: Monitor,
  light: Sun,
  dark: Moon,
};

/**
 * Single-icon theme control.
 *
 * Shows the mode that is currently selected — including "follow system", which
 * the stock two-icon switch could neither display nor return to — and cycles to
 * the next one on click.
 *
 * Until `next-themes` has read storage the button renders the default mode, so
 * the prerendered markup and the first client render agree.
 */
export interface ThemeToggleProps {
  /** Supplied by the layout slot that renders the control. */
  className?: string;
}

export default function ThemeToggle({ className = "" }: ThemeToggleProps) {
  const { setTheme, theme } = useTheme();
  const [isMounted, setIsMounted] = useState(false);
  const copy = useSiteCopy().theme;

  useEffect(() => setIsMounted(true), []);

  const mode = isMounted ? resolveThemeMode(theme) : DEFAULT_THEME_MODE;
  const Icon = MODE_ICONS[mode];
  // Assembled per locale in the copy: the control cycles rather than toggles,
  // which is not obvious from a single icon, so the hint travels with the name.
  const label = `${copy.modes[mode]} · ${copy.hint}`;

  function cycle() {
    const apply = () => setTheme(nextThemeMode(theme));

    // Matches the stock switch: a view transition keeps the palette swap from
    // reading as a flash on browsers that support it.
    if (typeof document.startViewTransition === "function") {
      document.startViewTransition(() => flushSync(apply));
    } else {
      apply();
    }
  }

  return (
    <button
      aria-label={label}
      className={`site-theme-toggle ${className}`.trim()}
      data-theme-mode={mode}
      data-theme-toggle=""
      onClick={cycle}
      title={label}
      type="button"
    >
      <Icon aria-hidden="true" />
    </button>
  );
}
