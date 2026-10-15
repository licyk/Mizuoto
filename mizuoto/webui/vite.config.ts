/// <reference types="vitest/config" />
import type { ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import vue from '@vitejs/plugin-vue';
import { createLogger, defineConfig, type ProxyOptions } from 'vite';

const backend = process.env.MIZUOTO_BACKEND ?? 'http://127.0.0.1:7869';

/**
 * The API server being down is normal while developing: it restarts whenever Python changes, and
 * the socket then reconnects every few seconds. Each attempt would otherwise print a stack trace
 * ("ws proxy error: read ECONNRESET"), which buries the output. These are reported as one short
 * line at most every few seconds; anything unexpected is still printed in full.
 */
const EXPECTED = new Set(['ECONNREFUSED', 'ECONNRESET', 'ECONNABORTED', 'EPIPE', 'ETIMEDOUT', 'EHOSTUNREACH']);
let lastNotice = 0;

/** Whether a proxy failure is the everyday "the API server is not up" kind, not a real fault. */
export function isExpectedProxyError(message: string, error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | undefined)?.code ?? '';
  return EXPECTED.has(code) && /proxy (socket )?error/.test(message);
}

function notice(code: string): void {
  if (Date.now() - lastNotice < 5000) return;
  lastNotice = Date.now();
  console.info(`[33m[api proxy][0m ${backend} is not answering (${code}). Start it with: mizuoto webui --no-open`);
}

/** Vite logs every proxy failure with a stack trace; the expected ones become a single line. */
const logger = createLogger();
const logError = logger.error.bind(logger);
logger.error = (message, options) => {
  if (isExpectedProxyError(message, options?.error)) {
    notice((options?.error as NodeJS.ErrnoException).code ?? '');
    return;
  }
  logError(message, options);
};

function apiProxy(ws = false): ProxyOptions {
  return {
    target: backend,
    ws,
    changeOrigin: false,
    configure(proxy) {
      proxy.on('error', (error, _request, target) => {
        const code = (error as NodeJS.ErrnoException).code ?? '';
        if (!EXPECTED.has(code)) return; // Vite's own handler reports it in full.
        notice(code);
        // Answer the request rather than leaving the browser waiting for a reply that never comes.
        const response = target as ServerResponse & Socket;
        if (response && 'writeHead' in response && !response.headersSent) {
          response.writeHead(502, { 'Content-Type': 'application/json' });
          response.end(JSON.stringify({ code: 'api_unreachable', message: `The API server at ${backend} is not answering.`, detail: { code } }));
        } else {
          response?.destroy?.();
        }
      });
      // A socket that dies mid-upgrade must not raise an unhandled error event.
      proxy.on('proxyReqWs', (_proxyRequest, _request, socket) => socket.on('error', () => undefined));
    },
  };
}

export default defineConfig({
  // Relative asset paths: the same build works at the root or under a reverse-proxy sub-path.
  base: './',
  customLogger: logger,
  plugins: [
    vue({
      template: {
        compilerOptions: {
          // @material/web elements are custom elements, not Vue components.
          isCustomElement: (tag) => tag.startsWith('md-'),
        },
      },
    }),
  ],
  resolve: {
    // Modules under src/ import each other through '@/...', never by a relative path.
    alias: { '@': new URL('./src', import.meta.url).pathname },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': apiProxy(true), // the browser-audio WebSocket is under /api
      '/openapi.json': apiProxy(),
      '/ws': apiProxy(true),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 5120,
  },
  test: {
    environment: 'happy-dom',
    setupFiles: ['src/test/setup.ts'],
    // material-color-utilities 0.4.0 has an import without a file extension that plain Node cannot resolve.
    server: { deps: { inline: ['@material/material-color-utilities'] } },
  },
});
