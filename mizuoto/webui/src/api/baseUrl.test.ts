import { describe, expect, it } from 'vitest';
import { deriveBaseUrl } from '@/api/baseUrl';

describe('deriveBaseUrl', () => {
  it('strips the assets folder at the root', () => {
    expect(deriveBaseUrl('https://example.com/assets/index-abc.js', 'https://example.com')).toBe('https://example.com');
  });
  it('keeps a reverse-proxy sub-path', () => {
    expect(deriveBaseUrl('https://example.com/hub/assets/index-abc.js', 'https://example.com')).toBe('https://example.com/hub');
  });
  it('uses the last assets segment', () => {
    expect(deriveBaseUrl('https://example.com/assets/x/assets/chunk.js', 'https://example.com')).toBe('https://example.com/assets/x');
  });
  it('falls back to the origin on the dev server', () => {
    expect(deriveBaseUrl('http://localhost:5173/src/api/baseUrl.ts', 'http://localhost:5173')).toBe('http://localhost:5173');
  });
  it('falls back to the page origin for a script from another origin', () => {
    expect(deriveBaseUrl('https://cdn.example.net/assets/a.js', 'http://127.0.0.1:7865')).toBe('http://127.0.0.1:7865');
  });
});
