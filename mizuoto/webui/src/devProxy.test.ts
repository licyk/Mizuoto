import { describe, expect, it } from 'vitest';
import { isExpectedProxyError } from '@/../vite.config';

/**
 * While developing, the API server restarts often and the socket reconnects. Those proxy
 * failures are reported as one short line; anything else must still be printed in full.
 */
describe('dev proxy errors', () => {
  const withCode = (code: string) => Object.assign(new Error('read ' + code), { code });

  it('treats a dropped or refused API connection as expected', () => {
    expect(isExpectedProxyError('ws proxy error:', withCode('ECONNRESET'))).toBe(true);
    expect(isExpectedProxyError('ws proxy socket error:', withCode('ECONNRESET'))).toBe(true);
    expect(isExpectedProxyError('http proxy error: /api/v1/settings', withCode('ECONNREFUSED'))).toBe(true);
    expect(isExpectedProxyError('ws proxy error:', withCode('EPIPE'))).toBe(true);
  });

  it('keeps everything else', () => {
    expect(isExpectedProxyError('ws proxy error:', withCode('EACCES'))).toBe(false);
    expect(isExpectedProxyError('ws proxy error:', undefined)).toBe(false);
    expect(isExpectedProxyError('Internal server error', withCode('ECONNRESET'))).toBe(false);
    expect(isExpectedProxyError('[vite] Pre-transform error: something broke', withCode('ECONNRESET'))).toBe(false);
  });
});
