import createClient, { type Middleware } from 'openapi-fetch';
import { BASE_URL } from '@/api/baseUrl';
import type { paths } from '@/api/schema';
import { useAuthStore } from '@/stores/auth';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly detail: Record<string, unknown> = {},
  ) {
    super(message);
  }

  /** Asset ids an ``asset_missing`` error names, so the interface can offer the download. */
  get missingAssets(): string[] {
    return this.code === 'asset_missing' && Array.isArray(this.detail.assets) ? (this.detail.assets as string[]) : [];
  }
}

const auth: Middleware = {
  onRequest({ request }) {
    const token = useAuthStore().token;
    if (token) request.headers.set('Authorization', `Bearer ${token}`);
    return request;
  },
  onResponse({ response }) {
    if (response.status === 401) useAuthStore().needed = true;
    return response;
  },
};

export const api = createClient<paths>({ baseUrl: BASE_URL });
api.use(auth);

type Result<T> = { data?: T; error?: unknown; response: Response };

export function toError(error: unknown, response: { status: number; statusText: string }): ApiError {
  if (error && typeof error === 'object') {
    const e = error as { code?: string; message?: string; detail?: unknown };
    if (typeof e.message === 'string') return new ApiError(response.status, e.code ?? 'error', e.message, (e.detail as Record<string, unknown>) ?? {});
    if (Array.isArray(e.detail)) {
      const first = e.detail[0] as { msg?: string; loc?: unknown[] } | undefined;
      return new ApiError(response.status, 'invalid_input', first?.msg ?? 'Invalid input', { errors: e.detail });
    }
  }
  return new ApiError(response.status, 'error', `${response.status} ${response.statusText}`);
}

/** Await an openapi-fetch call and return its data, or throw an ApiError with the server's message. */
export async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  const { data, error, response } = await call;
  if (!response.ok) throw toError(error, response);
  return data as T;
}

const V1 = `${BASE_URL}/api/v1`;

/** Media URLs. The token cookie authenticates <audio> requests; the server answers Range requests. */
export const urls = {
  audioFile: (id: string) => `${V1}/audio/files/${encodeURIComponent(id)}/content`,
  output: (id: string, download = false) => `${V1}/outputs/${encodeURIComponent(id)}/content${download ? '?download=true' : ''}`,
  outputSource: (id: string) => `${V1}/outputs/${encodeURIComponent(id)}/source`,
  modelArchive: (id: string) => `${V1}/models/${encodeURIComponent(id)}/archive`,
};

/** Save a blob under a name through a temporary link. */
export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Start a browser download of a URL, keeping the page. */
export function downloadUrl(url: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Download several outputs as one zip, streamed by the server. */
export async function downloadOutputsZip(ids: string[]): Promise<void> {
  const token = useAuthStore().token;
  const response = await fetch(`${V1}/outputs/zip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ ids }),
  });
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      /* not JSON */
    }
    throw toError(body, response);
  }
  saveBlob(await response.blob(), 'mizuoto-outputs.zip');
}

/**
 * Upload a file as the raw request body (``PUT``), reporting progress. fetch() cannot report
 * upload progress, so this uses XMLHttpRequest.
 */
export function uploadRaw<T>(path: string, query: Record<string, string>, file: Blob, onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `${BASE_URL}${path}?${new URLSearchParams(query)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    const token = useAuthStore().token;
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (e) => onProgress?.(e.loaded, e.lengthComputable ? e.total : file.size);
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        if (xhr.status === 401) useAuthStore().needed = true;
        reject(toError(body, { status: xhr.status, statusText: xhr.statusText }));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, 'network', 'Network error'));
    xhr.onabort = () => reject(new ApiError(0, 'aborted', 'Upload cancelled'));
    signal?.addEventListener('abort', () => xhr.abort());
    xhr.send(file);
  });
}
