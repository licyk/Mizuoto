import { generateScheme, ROLES, toKebab } from '@/theme/scheme';

export type ThemeMode = 'light' | 'dark' | 'system';

export interface ThemeOptions {
  mode: ThemeMode;
  sourceColor: string;
  contrast: number;
}

const darkQuery = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null;

export function resolveDark(mode: ThemeMode): boolean {
  return mode === 'dark' || (mode === 'system' && !!darkQuery?.matches);
}

/** Write the colour roles to :root as --md-sys-color-* custom properties. */
export function applyTheme(options: ThemeOptions): void {
  const isDark = resolveDark(options.mode);
  const scheme = generateScheme(options.sourceColor, isDark, options.contrast);
  const root = document.documentElement;
  for (const role of ROLES) {
    root.style.setProperty(`--md-sys-color-${toKebab(role)}`, scheme[role]);
  }
  root.dataset.theme = isDark ? 'dark' : 'light';
  root.style.colorScheme = isDark ? 'dark' : 'light';
}

/** Re-apply when the system preference changes and the mode is "system". */
export function watchSystemTheme(get: () => ThemeOptions): () => void {
  if (!darkQuery) return () => {};
  const listener = () => {
    if (get().mode === 'system') applyTheme(get());
  };
  darkQuery.addEventListener('change', listener);
  return () => darkQuery.removeEventListener('change', listener);
}
