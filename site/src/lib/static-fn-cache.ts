import { createMiddleware } from "@tanstack/react-start";
import { staticFunctionMiddleware } from "@tanstack/start-static-server-functions";

/**
 * `staticFunctionMiddleware` reads its prerendered answer from
 * `/__tsr/staticServerFnCache/<hash>.json`, an absolute path with no
 * `import.meta.env.BASE_URL` in it, while the files are published under the
 * site's prefix like every other asset. Deployed below a prefix the request
 * misses and the static host answers with the shell, which the client then
 * parses as JSON, so every page opened from a link comes up empty.
 *
 * This wraps it and points that one request at the prefix; served at the root
 * it does nothing.
 */
export const prefixedStaticFunctionMiddleware = createMiddleware({
  type: "function",
}).client(async (options) => {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, "");
  if (!base) return options.next();

  const fetchWithoutPrefix = globalThis.fetch;
  globalThis.fetch = (input, init) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    return fetchWithoutPrefix(
      url.startsWith("/__tsr/") ? `${base}${url}` : input,
      init,
    );
  };
  try {
    return await options.next();
  } finally {
    globalThis.fetch = fetchWithoutPrefix;
  }
});

/** The cache-reading middleware the documentation loader installs. */
export const loadPageMiddleware = [
  prefixedStaticFunctionMiddleware,
  staticFunctionMiddleware,
];
