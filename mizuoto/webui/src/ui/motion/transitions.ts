/** Named transitions and helpers for the motion system. See motion.css. */
import { ref, watch, type Ref } from 'vue';

export const TRANSITIONS = {
  fadeThrough: 'fade-through',
  sharedAxisX: 'shared-axis-x',
  container: 'container',
  sheet: 'sheet',
  scrim: 'scrim',
  list: 'list',
  collapse: 'collapse',
  snackbar: 'snackbar',
} as const;

const STAGGER_MS = 20;
const STAGGER_CAP = 12;

/** Style for the n-th item of a list transition: a 20 ms stagger, capped. */
export function staggerStyle(index: number): Record<string, string> {
  return { '--stagger': `${Math.min(index, STAGGER_CAP) * STAGGER_MS}ms` };
}

/**
 * The way a ``shared-axis-x`` tab change moves, for ``--axis-dir`` (as Hanaikada's tag tabs): 1 to a
 * tab on the right, -1 to one on the left. Set synchronously, before the panes re-render.
 */
export function useAxisDirection<T>(value: Ref<T>, order: () => readonly T[]): Ref<number> {
  const dir = ref(1);
  watch(value, (next, prev) => (dir.value = order().indexOf(next) >= order().indexOf(prev) ? 1 : -1), { flush: 'sync' });
  return dir;
}

/**
 * Whether a meter's value last went down, for audio-meter ballistics: a rise shows at once, a fall
 * settles over a longer transition. Set synchronously, with the value it describes.
 */
export function useFalling(value: () => number): Ref<boolean> {
  const falling = ref(false);
  watch(value, (next, prev) => (falling.value = next < prev), { flush: 'sync' });
  return falling;
}

export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  if (document.documentElement.dataset.motion === 'reduced') return true;
  return !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

/**
 * CSS variables that make a dialog grow from ``from`` (a card's rectangle) into ``to``.
 * Returns an empty object when the source rectangle is unknown, which falls back to fade and scale.
 */
export function containerFrom(from: DOMRect | null | undefined, to: DOMRect | null | undefined): Record<string, string> {
  if (!from || !to || to.width === 0 || to.height === 0) return {};
  const scaleX = from.width / to.width;
  const scaleY = from.height / to.height;
  const dx = from.left - to.left;
  const dy = from.top - to.top;
  return {
    '--from-origin': 'top left',
    '--from-transform': `translate(${dx}px, ${dy}px) scale(${scaleX}, ${scaleY})`,
  };
}

/**
 * Hooks for a ``list`` <TransitionGroup> of rows (its container ``position: relative``): the leaving
 * item is taken out of the flow (motion.css), so it is pinned where it was, at its own size, rather
 * than reflowing as a box without a width (its fields wrapping into one another) while it fades and
 * the others slide up.
 */
export const keepLeavingInPlace = {
  onBeforeLeave(el: Element) {
    const e = el as HTMLElement;
    Object.assign(e.style, { left: `${e.offsetLeft}px`, top: `${e.offsetTop}px`, width: `${e.offsetWidth}px`, height: `${e.offsetHeight}px` });
  },
};

/** Hooks for <Transition name="collapse"> that animate height from and to the content's size. */
export const collapseHooks = {
  onBeforeEnter(el: Element) {
    (el as HTMLElement).style.height = '0';
  },
  onEnter(el: Element) {
    const e = el as HTMLElement;
    e.style.height = `${e.scrollHeight}px`;
  },
  onAfterEnter(el: Element) {
    (el as HTMLElement).style.height = '';
  },
  onBeforeLeave(el: Element) {
    const e = el as HTMLElement;
    e.style.height = `${e.scrollHeight}px`;
  },
  onLeave(el: Element) {
    const e = el as HTMLElement;
    void e.offsetHeight;
    e.style.height = '0';
  },
};
