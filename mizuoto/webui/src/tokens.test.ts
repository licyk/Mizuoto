import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

/** No literal colours outside src/theme/: components use the colour-role tokens only. */
// Vitest runs from the web UI folder; import.meta.url is not a file URL under happy-dom.
const SRC = join(process.cwd(), 'src');
const HEX = /#[0-9a-fA-F]{3,8}\b/g;

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === 'theme' ? [] : files(path);
    return /\.(vue|css)$/.test(name) ? [path] : [];
  });
}

describe('design tokens', () => {
  it('has no hex colours outside src/theme', () => {
    const offenders = files(SRC).flatMap((path) =>
      (readFileSync(path, 'utf8').replace(/<script[\s\S]*?<\/script>/g, '').match(HEX) ?? []).map((hex) => `${relative(SRC, path)}: ${hex}`),
    );
    expect(offenders).toEqual([]);
  });
});
