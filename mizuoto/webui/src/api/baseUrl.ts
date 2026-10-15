/**
 * The deployment root, derived from the URL of the running script. No API URL is compiled in.
 *
 *   https://example.com/hub/assets/index-abc.js -> https://example.com/hub
 *   https://example.com/assets/index-abc.js     -> https://example.com
 *   http://localhost:5173/src/api/baseUrl.ts    -> http://localhost:5173   (dev server; the proxy forwards)
 *
 * Vite emits every chunk under <root>/assets/, so everything before the last "/assets/" is the root.
 * A script from another origin falls back to the page's origin.
 */
export function deriveBaseUrl(scriptUrl: string, pageOrigin: string): string {
  let url: URL;
  try {
    url = new URL(scriptUrl, pageOrigin);
  } catch {
    return pageOrigin;
  }
  if (url.origin !== pageOrigin) return pageOrigin;
  const index = url.pathname.lastIndexOf('/assets/');
  return url.origin + (index >= 0 ? url.pathname.slice(0, index) : '');
}

export const BASE_URL = deriveBaseUrl(import.meta.url, typeof window === 'undefined' ? 'http://localhost' : window.location.origin);
export const BASE_PATH = new URL(BASE_URL).pathname.replace(/\/$/, '');
