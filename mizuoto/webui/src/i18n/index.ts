import { computed, ref } from 'vue';
import en from '@/i18n/en';
import zhCN from '@/i18n/zh-CN';
import catalogZhCN from '@/i18n/catalog.zh-CN';
import { usePreferencesStore, type Locale, type LocalePreference } from '@/stores/preferences';

type Messages = typeof en;
const MESSAGES: Record<Locale, Messages> = { en, 'zh-CN': zhCN };
/** The server's own English (catalogue descriptions, parameter help) in each language, keyed by that English. */
const CATALOGS: Record<Locale, Record<string, string>> = { en: {}, 'zh-CN': catalogZhCN };
export const LOCALES: { value: Locale; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'zh-CN', label: '简体中文' },
];

/** The supported locale for a BCP 47 language tag, or English for one mizuoto has no translation for. */
export function detectLocale(tag: string | null | undefined): Locale {
  const lower = (tag ?? '').toLowerCase();
  if (lower === 'zh' || lower.startsWith('zh-')) return 'zh-CN';
  return 'en';
}

function systemTag(): string | undefined {
  return typeof navigator === 'undefined' ? undefined : (navigator.languages?.[0] ?? navigator.language);
}

const systemLocale = ref<Locale>(detectLocale(systemTag()));
if (typeof window !== 'undefined') window.addEventListener('languagechange', () => (systemLocale.value = detectLocale(systemTag())));

export function resolveLocale(preference: LocalePreference): Locale {
  return preference === 'auto' ? systemLocale.value : preference;
}

function lookup(tree: unknown, key: string): unknown {
  let node: unknown = tree;
  for (const part of key.split('.')) {
    if (node && typeof node === 'object' && part in (node as Record<string, unknown>)) node = (node as Record<string, unknown>)[part];
    else return undefined;
  }
  return node;
}

/** Whether a key exists, for messages chosen by a server value (an error code, a stage id). */
export function hasMessage(locale: Locale, key: string): boolean {
  return typeof lookup(MESSAGES[locale], key) === 'string' || typeof lookup(MESSAGES.en, key) === 'string';
}

export function translate(locale: Locale, key: string, params?: Record<string, string | number>): string {
  const found = lookup(MESSAGES[locale], key);
  const fallback = lookup(MESSAGES.en, key);
  const text = typeof found === 'string' ? found : typeof fallback === 'string' ? fallback : key;
  return params ? text.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? `{${name}}`)) : text;
}

/** Text the server sent in English (a model's description, a parameter's help) in ``locale``, when known. */
export function translateServerText(locale: Locale, text: string | null | undefined): string {
  if (!text) return '';
  return CATALOGS[locale][text] ?? text;
}

/** ``t(key, params)`` in the current locale; reactive to the locale preference and, on ``auto``, to the system's. */
export function useI18n() {
  const store = usePreferencesStore();
  const locale = computed(() => resolveLocale(store.prefs.locale));
  const t = (key: string, params?: Record<string, string | number>) => translate(locale.value, key, params);
  /** ``t`` for a key built from a server value, with a readable fallback when it has no message. */
  const tOr = (key: string, fallback: string, params?: Record<string, string | number>) => (hasMessage(locale.value, key) ? t(key, params) : fallback);
  /** ``tx(text)``: the server's English text (catalogue, parameter help) in the current locale, else as it is. */
  const tx = (text: string | null | undefined) => translateServerText(locale.value, text);
  const localeOptions = computed(() => [
    { value: 'auto', label: t('settings.languageAuto', { language: LOCALES.find((l) => l.value === systemLocale.value)?.label ?? '' }) },
    ...LOCALES,
  ]);
  return { t, tOr, tx, locale, localeOptions };
}
