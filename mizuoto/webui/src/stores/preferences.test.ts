import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, describe, expect, it } from 'vitest';
import { usePreferencesStore } from '@/stores/preferences';

const KEY = 'mizuoto:preferences';

function load(saved: object) {
  localStorage.setItem(KEY, JSON.stringify(saved));
  setActivePinia(createPinia());
  return usePreferencesStore().prefs;
}

describe('preferences', () => {
  beforeEach(() => localStorage.clear());

  it('starts with Animation following the system', () => {
    setActivePinia(createPinia());
    expect(usePreferencesStore().prefs.motion).toBe('system');
  });

  it('puts Animation back to following the system once, for preferences saved before version 2', () => {
    const prefs = load({ motion: 'reduced', theme: 'dark' });
    expect(prefs.motion).toBe('system');
    expect(prefs.theme).toBe('dark');
    expect(prefs.version).toBe(2);
  });

  it('keeps Reduced when it was chosen since', () => {
    expect(load({ motion: 'reduced', version: 2 }).motion).toBe('reduced');
  });
});
