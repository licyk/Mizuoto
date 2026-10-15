/** Formatting helpers shared by views. */

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let i = 0;
  while (Math.abs(value) >= 1024 && i < units.length - 1) {
    value /= 1024;
    i++;
  }
  return i === 0 ? `${Math.round(value)} B` : `${value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`;
}

export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function formatDate(value: string | null | undefined, locale: string): string {
  if (!value) return '—';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString(locale);
}

export function formatEta(remaining: number, speed: number): string {
  if (!speed || remaining <= 0) return '';
  const s = Math.round(remaining / speed);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}




/** A media length: ``0:07``, ``12:30``, ``1:02:03``. */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

/** A sample rate as kHz: ``48 kHz``, ``44.1 kHz``. */
export function formatRate(hz: number | null | undefined): string {
  if (!hz) return '—';
  const k = hz / 1000;
  return `${Number.isInteger(k) ? k : k.toFixed(1)} kHz`;
}

/** Megabytes from the server's MiB figures: ``1.2 GB`` or ``840 MB``. */
export function formatMiB(mb: number | null | undefined): string {
  if (mb === null || mb === undefined) return '—';
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/** Relative time for "last activity": ``2 min ago``; falls back to the date. */
export function formatAgo(value: string | null | undefined, locale: string): string {
  if (!value) return '—';
  const d = new Date(value);
  const s = (Date.now() - d.getTime()) / 1000;
  if (Number.isNaN(s)) return '—';
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  if (s < 60) return rtf.format(-Math.round(s), 'second');
  if (s < 3600) return rtf.format(-Math.round(s / 60), 'minute');
  if (s < 86400) return rtf.format(-Math.round(s / 3600), 'hour');
  return rtf.format(-Math.round(s / 86400), 'day');
}
