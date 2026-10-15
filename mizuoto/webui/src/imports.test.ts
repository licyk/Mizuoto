import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

/** Modules under src/ import each other through the '@/' alias, never by a relative path. */
// Vitest runs from the web UI folder; import.meta.url is not a file URL under happy-dom.
const SRC = join(process.cwd(), 'src');
const RELATIVE = /\b(?:from|import)\s*\(?\s*['"](\.{1,2}\/[^'"]*)['"]/g;

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return files(path);
    return /\.(ts|vue|css)$/.test(name) ? [path] : [];
  });
}

describe('module imports', () => {
  it('uses the @/ alias, with no relative path', () => {
    const offenders = files(SRC).flatMap((path) =>
      [...readFileSync(path, 'utf8').matchAll(RELATIVE)].map((m) => `${relative(SRC, path)}: ${m[1]}`),
    );
    expect(offenders).toEqual([]);
  });
});
