import { reactive } from 'vue';

export interface SnackbarMessage {
  id: number;
  text: string;
  actionLabel?: string;
  action?: () => void;
  error?: boolean;
  timeout: number;
}

const state = reactive<{ queue: SnackbarMessage[] }>({ queue: [] });
let nextId = 1;

/** A queue of snackbars, one shown at a time. */
export function useSnackbar() {
  function show(text: string, options: { actionLabel?: string; action?: () => void; error?: boolean; timeout?: number } = {}) {
    state.queue.push({ id: nextId++, text, timeout: options.timeout ?? (options.actionLabel ? 6000 : 4000), ...options });
  }
  function dismiss(id: number) {
    const i = state.queue.findIndex((m) => m.id === id);
    if (i >= 0) state.queue.splice(i, 1);
  }
  return { state, show, error: (text: string) => show(text, { error: true, timeout: 7000 }), dismiss };
}
