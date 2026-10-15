import { defineStore } from 'pinia';
import { reactive, watch } from 'vue';
import { getClientState, putClientState } from '@/api/queries/app';
import type { ThemeMode } from '@/theme/applyTheme';
import { DEFAULT_SOURCE_COLOR } from '@/theme/scheme';

export type Locale = 'en' | 'zh-CN';
/** A locale, or ``auto`` to follow the system's language. */
export type LocalePreference = Locale | 'auto';
const LOCALE_PREFERENCES: LocalePreference[] = ['auto', 'en', 'zh-CN'];
export type MotionPreference = 'system' | 'reduced';

export interface Preferences {
  theme: ThemeMode;
  sourceColor: string;
  contrast: number;
  locale: LocalePreference;
  /** "Reduced" applies the reduced-motion rules whatever the system says (Settings › Appearance). */
  motion: MotionPreference;
  lastScreen: string;
  /** The last model and library voice chosen in Speak, so they are selected again next time. */
  speakModel: string | null;
  speakVoice: string | null;
  /** The language last chosen in Speak (``auto``: detect). */
  speakLanguage: string;
  /** Speak plays the speech as it is made. */
  speakStream: boolean;
  /** The shape of saved preferences; older ones are migrated on load (``merge``). */
  version: number;
}

const STORAGE_KEY = 'mizuoto:preferences';
const SERVER_KEY = 'preferences';
/** 2: Animation follows the system again; saved preferences before it had "Reduced" stuck on. */
const PREFERENCES_VERSION = 2;

export const DEFAULTS: Preferences = {
  theme: 'system',
  sourceColor: DEFAULT_SOURCE_COLOR,
  contrast: 0,
  locale: 'auto',
  motion: 'system',
  lastScreen: 'speak',
  speakModel: null,
  speakVoice: null,
  speakLanguage: 'auto',
  speakStream: false,
  version: PREFERENCES_VERSION,
};

function readLocal(): Partial<Preferences> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}');
  } catch {
    return {};
  }
}

function merge(base: Preferences, patch: Partial<Preferences>): Preferences {
  const merged = { ...base, ...patch };
  // Saved before version 2: Animation goes back to its default, following the system.
  if ((patch.version ?? 1) < 2) merged.motion = DEFAULTS.motion;
  merged.version = PREFERENCES_VERSION;
  // A locale this build does not know (saved by a newer one, or edited by hand) follows the system.
  if (!LOCALE_PREFERENCES.includes(merged.locale)) merged.locale = 'auto';
  return merged;
}

/**
 * Interface preferences are client state. They persist to the server's client-state endpoint, so
 * they follow the user across browsers, with a localStorage copy so index.html can apply the theme
 * before the bundle loads.
 */
export const usePreferencesStore = defineStore('preferences', () => {
  const prefs = reactive<Preferences>(merge(DEFAULTS, readLocal()));
  let serverTimer: ReturnType<typeof setTimeout> | undefined;
  let loaded = false;

  watch(
    prefs,
    (value) => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
      } catch {
        /* private mode */
      }
      if (!loaded) return;
      clearTimeout(serverTimer);
      serverTimer = setTimeout(() => putClientState(SERVER_KEY, { ...value }).catch(() => undefined), 500);
    },
    { deep: true },
  );

  async function loadFromServer() {
    try {
      const remote = await getClientState<Partial<Preferences>>(SERVER_KEY);
      if (remote && typeof remote === 'object') Object.assign(prefs, merge(prefs, remote));
    } catch {
      /* offline or needs a token: keep local values */
    }
    loaded = true;
  }

  return { prefs, loadFromServer };
});
