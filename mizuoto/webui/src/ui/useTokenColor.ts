/** Read colour roles from the theme for canvas drawing, which cannot use CSS variables directly. */
export function tokenColor(el: Element | null, name: string, fallback = 'currentColor'): string {
  if (!el || typeof getComputedStyle === 'undefined') return fallback;
  const value = getComputedStyle(el).getPropertyValue(name).trim();
  return value || fallback;
}
