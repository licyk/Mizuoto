import { nextTick, ref, type Ref } from 'vue';
import type { RouteLocationNormalized, Router } from 'vue-router';
import { prefersReducedMotion } from '@/ui/motion/transitions';

interface ViewTransition {
  finished: Promise<void>;
  ready?: Promise<void>;
  updateCallbackDone?: Promise<void>;
}
type StartViewTransition = (cb: () => Promise<void> | void) => ViewTransition;

/** The ``view-transition-name`` of the element pair a navigation morphs. */
const MORPH = 'morph';
/** The key of the element pair the current navigation morphs (``morph`` option), or null. */
export const morphKey = ref<string | null>(null);

/**
 * Style for an element that morphs into its counterpart on the other screen (a card and the page it
 * opens: the container transform) while ``key`` is the navigation's morph key. Exactly one element
 * per screen may carry a key, or the browser skips the transition.
 */
export function morphStyle(key: string): Record<string, string> {
  return morphKey.value === key ? { viewTransitionName: MORPH, viewTransitionGroup: 'nearest' } : {};
}

export interface ViewTransitionOptions {
  /** The key of the elements that morph from ``from`` into ``to`` (see ``morphStyle``), or null. */
  morph?: (to: RouteLocationNormalized, from: RouteLocationNormalized) => string | null;
}

/**
 * Wrap router navigation between destinations in ``document.startViewTransition`` where the browser
 * has it. The returned ref is true then, and the app renders its views without its own
 * fade-through ``<Transition>``; without the API that ``<Transition>`` is the fallback.
 *
 * The browser pauses rendering, and with it ``requestAnimationFrame``, while the update callback
 * runs, so the callback must finish on Vue's ``nextTick`` (the new view is in the DOM then), never on
 * a frame. Navigation inside one destination (a query change) is not wrapped. Reduced motion is
 * still wrapped: motion.css turns the fade through into a short crossfade, never none.
 *
 * With ``morph``, the pair it names is set before the old state is captured (Vue renders on a
 * microtask, before the browser's next frame) and kept until the transition ends, so the same rule
 * morphs the card into the page and the page back into the card. Not under reduced motion.
 */
export function installViewTransitions(router: Router, options: ViewTransitionOptions = {}): Ref<boolean> {
  const doc = typeof document === 'undefined' ? null : (document as Document & { startViewTransition?: StartViewTransition });
  const supported = ref(!!doc?.startViewTransition);
  if (!doc?.startViewTransition) return supported;
  const start = doc.startViewTransition.bind(doc);
  let finish: (() => void) | null = null;
  let latest = 0;

  router.beforeResolve((to, from) => {
    if (!from.name || to.name === from.name) return true;
    // A navigation that interrupts an unfinished one lets the earlier transition end now.
    finish?.();
    const key = prefersReducedMotion() ? null : (options.morph?.(to, from) ?? null);
    morphKey.value = key;
    const id = ++latest;
    return new Promise<boolean>((resolve) => {
      const vt = start(
        () =>
          new Promise<void>((done) => {
            finish = () => {
              finish = null;
              done();
            };
            resolve(true);
          }),
      );
      // A transition skipped by a newer one rejects these; that is expected, not an error.
      for (const p of [vt.finished, vt.ready, vt.updateCallbackDone]) p?.catch(() => undefined);
      // The names go once it ends, unless a newer navigation has set its own.
      vt.finished
        .catch(() => undefined)
        .then(() => {
          if (id === latest) morphKey.value = null;
        });
    });
  });
  router.afterEach(() => {
    if (finish) nextTick().then(() => finish?.());
  });
  router.onError(() => finish?.());
  return supported;
}
