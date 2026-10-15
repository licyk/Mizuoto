import { computed, nextTick, onBeforeUnmount, readonly, ref, shallowReactive, watch, type Ref } from 'vue';

/**
 * The one record of what is open over the page — dialogs, sheets, menus, an open picker or select —
 * in the order it opened, and the rules every layer keeps because of that order:
 *
 * - **Painting.** A layer opened later paints above every layer opened before it. Each kind keeps
 *   a floor (``LAYER_Z``), so a menu still clears a dialog opened beside it, but two dialogs, or a
 *   sheet opened from a dialog, stack by when they opened, never by where their component sits in
 *   a template. ``inline`` layers (a picker listing in place, a native-popover select) take part in
 *   the order but are drawn by the page: their z-index stays a CSS token.
 * - **Keyboard.** Escape closes the topmost layer alone; ``isTop()`` tells a layer beneath to
 *   leave its own keys (Tab, arrows) alone until it is on top again.
 * - **Outside clicks.** A click outside the topmost layer closes it alone, not the dialog under the
 *   menu as well. The press must start outside too, so a text selection dragged out of a dialog
 *   does not close it. A press on ``[data-layer-ignore]`` (the snackbar) is never outside.
 * - **Modality.** While a ``modal`` layer is open, the page (``useLayerBase``) and every layer under
 *   it are ``inert``: no clicks, no Tab, hidden from assistive technology. Layers above it (a menu
 *   opened from the dialog) and the snackbar stay live.
 * - **Focus.** A layer focuses ``initialFocus`` once it has rendered, and gives focus back to what
 *   had it when it opened, unless the user has meanwhile put focus somewhere else. A layer opened
 *   from a menu item (which the menu takes away as it closes) returns focus to the menu's trigger.
 */

/** Floors of the layers drawn over the page. The in-page stacking (dropdown, sticky footer) stays in ``--app-z-*``. */
export const LAYER_Z = { sheet: 100, dialog: 200, snackbar: 300, menu: 400 } as const;

export type LayerKind = 'inline' | 'sheet' | 'dialog' | 'menu';

export interface LayerOptions {
  kind: LayerKind;
  /** Whether the layer is shown. */
  open: () => boolean;
  /** Close it: Escape or a click outside while it is the topmost layer. */
  close: () => void;
  /** What belongs to the layer, its trigger included: a press inside them is not outside. A lower layer's are made inert under a modal. */
  elements?: () => (Element | null | undefined)[];
  /** Make the page and the layers under it inert while this one is open. */
  modal?: boolean;
  /** Focused once the layer has rendered. */
  initialFocus?: () => HTMLElement | null | undefined;
}

interface Layer extends LayerOptions {
  opener: Element | null;
}

const stack = shallowReactive<Layer[]>([]);
const base = ref<HTMLElement | null>(null);
/** Layers closed in this tick, so one opened from them can find who opened them. */
let justClosed: { elements: Element[]; opener: Element | null }[] = [];
let inerted = new Set<Element>();
let pressedOutside: Layer | null = null;

const top = (): Layer | undefined => stack[stack.length - 1];
const elementsOf = (layer: Layer): Element[] => (layer.elements?.() ?? []).filter((el): el is Element => !!el);

const zIndexes = computed(() => {
  const z = new Map<Layer, number>();
  let below = 0;
  for (const layer of stack) {
    if (layer.kind === 'inline') continue;
    below = Math.max(LAYER_Z[layer.kind], below + 1);
    z.set(layer, below);
  }
  return z;
});

/** The snackbar: over every modal layer, so a message about what a dialog did shows over its scrim. */
export const snackbarZ = computed(() => {
  let z: number = LAYER_Z.snackbar;
  for (const layer of stack) if (layer.modal) z = Math.max(z, (zIndexes.value.get(layer) ?? 0) + 1);
  return z;
});

function isInside(layer: Layer, event: Event): boolean {
  const path = event.composedPath();
  const own = elementsOf(layer);
  return path.some((node) => own.includes(node as Element) || (node instanceof HTMLElement && node.dataset.layerIgnore !== undefined));
}

function onKey(event: KeyboardEvent) {
  // Something inside (a select's own menu) has already dealt with it.
  if (event.key !== 'Escape' || event.defaultPrevented || !stack.length) return;
  event.preventDefault();
  top()!.close();
}

function onPointerDown(event: PointerEvent) {
  const layer = top();
  pressedOutside = layer && !isInside(layer, event) ? layer : null;
}

function onClick(event: MouseEvent) {
  const layer = top();
  const pressed = pressedOutside;
  pressedOutside = null;
  // A click from the keyboard has no press: it never dismisses.
  if (layer && pressed === layer && !isInside(layer, event)) layer.close();
}

function listen(on: boolean) {
  if (on) {
    document.addEventListener('keydown', onKey);
    // Capture: the layer under the topmost one must not act on the click first.
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('click', onClick, true);
  } else {
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('pointerdown', onPointerDown, true);
    document.removeEventListener('click', onClick, true);
  }
}

/** Make the page and every layer under the topmost modal one inert; undo it for the rest. */
function applyInert() {
  const next = new Set<Element>();
  let modal = -1;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (stack[i].modal) {
      modal = i;
      break;
    }
  }
  if (modal >= 0) {
    if (base.value) next.add(base.value);
    for (const layer of stack.slice(0, modal)) for (const el of elementsOf(layer)) next.add(el);
  }
  for (const el of inerted) if (!next.has(el)) el.removeAttribute('inert');
  for (const el of next) el.setAttribute('inert', '');
  inerted = next;
}

function push(layer: Layer) {
  // Opened from a layer closing in this same tick (a menu item): what opened that one is the opener.
  let opener = document.activeElement;
  for (const closed of justClosed) if (opener && closed.elements.some((el) => el.contains(opener))) opener = closed.opener;
  layer.opener = opener;
  if (!stack.length) listen(true);
  stack.push(layer);
  applyInert();
  void nextTick(() => {
    if (stack.includes(layer)) layer.initialFocus?.()?.focus();
  });
}

function remove(layer: Layer) {
  const i = stack.indexOf(layer);
  if (i < 0) return;
  // Taken now: a ref to the layer's panel is gone once it has rendered closed.
  const own = elementsOf(layer);
  stack.splice(i, 1);
  if (!stack.length) {
    listen(false);
    pressedOutside = null;
  }
  applyInert();
  const record = { elements: own, opener: layer.opener };
  justClosed.push(record);
  void nextTick(() => {
    justClosed = justClosed.filter((r) => r !== record);
    const active = document.activeElement;
    const lost = !active || active === document.body || own.some((el) => el.contains(active));
    const opener = layer.opener as HTMLElement | null;
    if (lost && opener?.isConnected && !opener.closest('[inert]')) opener.focus?.();
  });
}

/** Register a layer shown while ``open()`` holds. Its ``zIndex`` keeps the last value while it animates out. */
export function useLayer(options: LayerOptions): { isTop: () => boolean; zIndex: Readonly<Ref<number>> } {
  const layer: Layer = { ...options, opener: null };
  const zIndex = ref(options.kind === 'inline' ? 0 : LAYER_Z[options.kind]);
  // Synchronous, so a layer closed by a key is off the stack before the next listener asks.
  watch(
    options.open,
    (value) => {
      remove(layer);
      if (value) push(layer);
    },
    { immediate: true, flush: 'sync' },
  );
  watch(
    () => zIndexes.value.get(layer),
    (z) => {
      if (z !== undefined) zIndex.value = z;
    },
    { immediate: true, flush: 'sync' },
  );
  onBeforeUnmount(() => remove(layer));
  return { isTop: () => top() === layer, zIndex: readonly(zIndex) };
}

/** The page every layer opens over (the app shell): inert while a modal layer is open. */
export function useLayerBase(el: Ref<HTMLElement | null>): void {
  watch(
    el,
    (value) => {
      base.value?.removeAttribute('inert');
      base.value = value;
      applyInert();
    },
    { immediate: true, flush: 'post' },
  );
  onBeforeUnmount(() => {
    if (base.value === el.value) {
      base.value?.removeAttribute('inert');
      base.value = null;
    }
  });
}

/** Whether any layer is open over the page. */
export const layerOpen = (): boolean => stack.length > 0;
