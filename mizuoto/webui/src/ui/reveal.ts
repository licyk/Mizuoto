import { nextTick, onBeforeUnmount, ref, watch, type Ref } from 'vue';
import { prefersReducedMotion } from '@/ui/motion/transitions';

const MARGIN = 8;

function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p; p = p.parentElement) {
    const overflow = getComputedStyle(p).overflowY;
    if (overflow === 'auto' || overflow === 'scroll') return p;
  }
  return null;
}

/**
 * Scroll a dropdown that lists in place (the voice picker, a device menu) fully into view, clear
 * of the page's sticky footer (``data-page-footer``): a list opened near the bottom moves up
 * instead of hiding under the Start bar or covering it, and one opened over its field and cut off
 * at the top moves down. A list taller than the room shows from its top.
 */
export function revealDropdown(el: HTMLElement | null | undefined): void {
  if (!el) return;
  const box = scrollParent(el);
  if (!box) return;
  const view = box.getBoundingClientRect();
  const footer = box.querySelector<HTMLElement>('[data-page-footer]');
  const bottom = footer ? Math.min(view.bottom, footer.getBoundingClientRect().top) : view.bottom;
  const rect = el.getBoundingClientRect();
  // A list cut off at the top (one opened over its field) moves down; one under the footer, up.
  const delta = rect.top < view.top + MARGIN ? rect.top - view.top - MARGIN : Math.max(0, Math.min(rect.bottom + MARGIN - bottom, rect.top - view.top - MARGIN));
  if (delta !== 0) box.scrollBy({ top: delta, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
}

/** Where a dropdown that lists in place goes: under its field or over it, and how much room the page must add. */
export interface Placement {
  above: boolean;
  /** Pixels the page must grow by at its end for the list to fit under the field. */
  reserve: number;
}

/**
 * Place a dropdown of ``height`` pixels that lists in place under ``anchor``, within the page that
 * holds the sticky footer. A sticky footer stays within its page: a list hanging past the page's end
 * lets the scroller scroll past it too, and the footer, stopped at the page's end, rises off the
 * bottom. So a list with no room under its field opens over it when the room is there, and else
 * the page makes the room it lacks (``reserve``).
 */
export function placeDropdown(anchor: HTMLElement, height: number): Placement {
  const box = scrollParent(anchor);
  const footer = box?.querySelector<HTMLElement>('[data-page-footer]');
  const page = footer?.parentElement;
  if (!footer || !page || !page.contains(anchor)) return { above: false, reserve: 0 };
  const pageRect = page.getBoundingClientRect();
  const rect = anchor.getBoundingClientRect();
  // The page's content ends where the footer starts in the flow.
  const below = pageRect.bottom - footer.offsetHeight - rect.bottom - MARGIN;
  const above = rect.top - pageRect.top - MARGIN;
  if (height <= below) return { above: false, reserve: 0 };
  if (height <= above) return { above: true, reserve: 0 };
  return { above: false, reserve: Math.ceil(height - below) };
}

/** Let the page holding ``anchor`` grow by ``px`` at its end (0: as it was), before its sticky footer. */
function reserveRoom(anchor: HTMLElement, px: number): () => void {
  const footer = scrollParent(anchor)?.querySelector<HTMLElement>('[data-page-footer]');
  const content = footer?.previousElementSibling as HTMLElement | null | undefined;
  if (!content || px <= 0) return () => undefined;
  const before = content.style.marginBottom;
  content.style.marginBottom = `${px}px`;
  return () => (content.style.marginBottom = before);
}

/**
 * Keep a dropdown revealed while it is open: on opening, and again whenever its size changes (the
 * search field inside is a web component that lays out a moment later; a search shortens the list).
 * With ``anchor`` (its field) it is also placed (``placeDropdown``): ``above`` says to open it over
 * the field, and the room a list needs at the page's end is made while it is open.
 */
export function useRevealWhileOpen(open: () => boolean, el: Ref<HTMLElement | null>, anchor?: Ref<HTMLElement | null>): { above: Ref<boolean> } {
  const above = ref(false);
  let observer: ResizeObserver | null = null;
  let release: () => void = () => undefined;
  const stop = () => {
    observer?.disconnect();
    observer = null;
    release();
    release = () => undefined;
  };
  const place = (target: HTMLElement) => {
    const a = anchor?.value;
    if (a) {
      release();
      const p = placeDropdown(a, target.offsetHeight + 4);
      above.value = p.above;
      release = reserveRoom(a, p.reserve);
    }
    revealDropdown(target);
  };
  watch(open, async (isOpen) => {
    stop();
    if (!isOpen) return;
    await nextTick();
    const target = el.value;
    if (!target) return;
    place(target);
    if (typeof ResizeObserver === 'undefined') return;
    observer = new ResizeObserver(() => place(target));
    observer.observe(target);
  });
  onBeforeUnmount(stop);
  return { above };
}
