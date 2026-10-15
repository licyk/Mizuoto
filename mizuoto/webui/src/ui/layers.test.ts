import { mount, type VueWrapper } from '@vue/test-utils';
import { afterEach, describe, expect, it } from 'vitest';
import { defineComponent, h, nextTick, ref, type PropType, type Ref } from 'vue';
import AppDialog from '@/ui/AppDialog.vue';
import AppMenu from '@/ui/AppMenu.vue';
import { LAYER_Z, layerOpen, snackbarZ, useLayer, useLayerBase, type LayerKind } from '@/ui/layers';

interface LayerVm {
  open: boolean;
  isTop: () => boolean;
  zIndex: number;
  el: HTMLElement;
}

/** A layer whose panel holds one button; open from the start unless ``closed``. */
const Layer = defineComponent({
  props: { kind: { type: String as PropType<LayerKind>, default: 'dialog' }, modal: Boolean, closed: Boolean },
  setup(props, { expose }) {
    const open = ref(!props.closed);
    const el = ref<HTMLElement | null>(null);
    const layer = useLayer({
      kind: props.kind,
      modal: props.modal,
      open: () => open.value,
      close: () => (open.value = false),
      elements: () => [el.value],
      initialFocus: () => el.value?.querySelector('button'),
    });
    expose({ open, isTop: layer.isTop, zIndex: layer.zIndex, el });
    return () => h('div', { ref: el, class: 'panel' }, [h('button', { type: 'button' }, 'inside')]);
  },
});

const Page = defineComponent({
  setup(_, { expose }) {
    const el: Ref<HTMLElement | null> = ref(null);
    useLayerBase(el);
    expose({ el });
    return () => h('div', { ref: el, class: 'page' }, [h('button', { type: 'button', class: 'opener' }, 'open')]);
  },
});

const mounted: VueWrapper[] = [];
function layer(props: { kind?: LayerKind; modal?: boolean; closed?: boolean } = {}): LayerVm {
  const w = mount(Layer, { props, attachTo: document.body });
  mounted.push(w);
  return w.vm as unknown as LayerVm;
}
function page(): HTMLElement {
  const w = mount(Page, { attachTo: document.body });
  mounted.push(w);
  return (w.vm as unknown as { el: HTMLElement }).el;
}

const escape = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
/** A press and a release on ``down`` and ``up``; the click lands on ``up``, as when they share a target. */
function click(down: Element, up: Element = down) {
  down.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, composed: true }));
  up.dispatchEvent(new MouseEvent('click', { bubbles: true, composed: true }));
}

afterEach(() => {
  while (mounted.length) mounted.pop()!.unmount();
  document.body.innerHTML = '';
  expect(layerOpen()).toBe(false);
});

describe('layers: keyboard', () => {
  it('closes only the topmost layer on Escape, the one under it next', () => {
    const sheet = layer({ kind: 'sheet', modal: true });
    const dialog = layer({ modal: true });
    expect([sheet.isTop(), dialog.isTop()]).toEqual([false, true]);
    escape();
    expect([sheet.open, dialog.open]).toEqual([true, false]);
    expect(sheet.isTop()).toBe(true);
    escape();
    expect(sheet.open).toBe(false);
  });

  it('leaves Escape to a layer that has already handled it', () => {
    const dialog = layer({ modal: true });
    const event = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    event.preventDefault();
    document.dispatchEvent(event);
    expect(dialog.open).toBe(true);
  });

  it('leaves the stack when unmounted open', () => {
    mount(Layer).unmount();
    expect(layerOpen()).toBe(false);
  });
});

describe('layers: painting', () => {
  it('paints a layer opened later above every earlier one, whatever its kind', () => {
    const dialog = layer({ modal: true });
    const sheet = layer({ kind: 'sheet', modal: true });
    const nested = layer({ modal: true });
    expect(dialog.zIndex).toBe(LAYER_Z.dialog);
    // A sheet opened from a dialog: its floor is lower, the order wins.
    expect(sheet.zIndex).toBe(LAYER_Z.dialog + 1);
    expect(nested.zIndex).toBe(LAYER_Z.dialog + 2);
  });

  it('keeps each kind over its floor and inline layers out of the count', () => {
    const sheet = layer({ kind: 'sheet', modal: true });
    layer({ kind: 'inline' });
    const menu = layer({ kind: 'menu' });
    expect(sheet.zIndex).toBe(LAYER_Z.sheet);
    expect(menu.zIndex).toBe(LAYER_Z.menu);
  });

  it('keeps a closing layer where it was while it animates out', async () => {
    layer({ modal: true });
    const nested = layer({ modal: true });
    nested.open = false;
    await nextTick();
    expect(nested.zIndex).toBe(LAYER_Z.dialog + 1);
  });

  it('puts the snackbar over every modal layer', () => {
    expect(snackbarZ.value).toBe(LAYER_Z.snackbar);
    layer({ modal: true });
    layer({ kind: 'menu' });
    expect(snackbarZ.value).toBe(LAYER_Z.snackbar);
    const sheets = Array.from({ length: LAYER_Z.snackbar - LAYER_Z.dialog }, () => layer({ modal: true }));
    expect(snackbarZ.value).toBe(sheets[sheets.length - 1].zIndex + 1);
  });
});

describe('layers: clicks outside', () => {
  it('closes only the topmost layer, the one under it on the next click', () => {
    const dialog = layer({ modal: true });
    const menu = layer({ kind: 'menu' });
    click(document.body);
    expect([dialog.open, menu.open]).toEqual([true, false]);
    click(document.body);
    expect(dialog.open).toBe(false);
  });

  it('ignores a click inside the layer, one dragged out of it, a keyboard click and the snackbar', () => {
    const dialog = layer({ modal: true });
    click(dialog.el.querySelector('button')!);
    click(dialog.el, document.body);
    document.body.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const snackbar = document.createElement('div');
    snackbar.dataset.layerIgnore = '';
    snackbar.append(document.createElement('button'));
    document.body.append(snackbar);
    click(snackbar.firstElementChild!);
    expect(dialog.open).toBe(true);
  });

  it('a click inside a lower layer closes the one over it, not the lower one', () => {
    const dialog = layer({ modal: true });
    const menu = layer({ kind: 'menu' });
    click(dialog.el.querySelector('button')!);
    expect([dialog.open, menu.open]).toEqual([true, false]);
  });
});

describe('layers: modality', () => {
  it('makes the page and the layers under a modal inert, and not what opened over it', async () => {
    const base = page();
    const first = layer({ modal: true });
    expect(base.hasAttribute('inert')).toBe(true);
    expect(first.el.hasAttribute('inert')).toBe(false);
    const second = layer({ kind: 'sheet', modal: true });
    const menu = layer({ kind: 'menu' });
    expect(first.el.hasAttribute('inert')).toBe(true);
    expect(second.el.hasAttribute('inert')).toBe(false);
    expect(menu.el.hasAttribute('inert')).toBe(false);
    second.open = false;
    await nextTick();
    expect(first.el.hasAttribute('inert')).toBe(false);
    expect(base.hasAttribute('inert')).toBe(true);
    first.open = false;
    await nextTick();
    expect(base.hasAttribute('inert')).toBe(false);
  });

  it('leaves the page live under a menu alone', () => {
    const base = page();
    layer({ kind: 'menu' });
    expect(base.hasAttribute('inert')).toBe(false);
  });
});

describe('layers: focus', () => {
  it('focuses the layer, then gives focus back to what opened it', async () => {
    const opener = page().querySelector<HTMLButtonElement>('.opener')!;
    opener.focus();
    const dialog = layer({ modal: true });
    await nextTick();
    expect(document.activeElement).toBe(dialog.el.querySelector('button'));
    dialog.open = false;
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(opener);
  });

  it('leaves focus where the user put it meanwhile', async () => {
    const base = page();
    const opener = base.querySelector<HTMLButtonElement>('.opener')!;
    const other = document.createElement('button');
    base.append(other);
    opener.focus();
    const menu = layer({ kind: 'menu' });
    await nextTick();
    menu.open = false;
    other.focus();
    await nextTick();
    expect(document.activeElement).toBe(other);
  });

  it("returns focus from a dialog opened by a menu item to the menu's trigger", async () => {
    const opener = page().querySelector<HTMLButtonElement>('.opener')!;
    opener.focus();
    const menu = layer({ kind: 'menu' });
    await nextTick();
    // The item that chose the dialog has focus as the menu closes and the dialog opens.
    menu.open = false;
    const dialog = layer({ modal: true });
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(dialog.el.querySelector('button'));
    dialog.open = false;
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(opener);
  });
});

describe('layers: components', () => {
  it('a click on the scrim with a menu open in a dialog closes the menu, the next one the dialog', async () => {
    const w = mount(
      defineComponent({
        setup() {
          const open = ref(true);
          return () =>
            h(AppDialog, { open: open.value, 'onUpdate:open': (v: boolean) => (open.value = v), title: 'Dialog' }, () =>
              h(AppMenu, { items: [{ id: 'a', label: 'A' }] }, { default: ({ toggle }: { toggle: () => void }) => h('button', { class: 'trigger', onClick: toggle }, 'menu') }),
            );
        },
      }),
      { attachTo: document.body },
    );
    mounted.push(w);
    await nextTick();
    const scrim = document.querySelector('.scrim')!;
    const layerEl = document.querySelector<HTMLElement>('.layer')!;
    expect(Number(layerEl.style.zIndex)).toBe(LAYER_Z.dialog);
    document.querySelector<HTMLButtonElement>('.trigger')!.click();
    await nextTick();
    const menu = document.querySelector<HTMLElement>('[role="menu"]')!;
    expect(Number(menu.style.zIndex)).toBe(LAYER_Z.menu);
    click(scrim);
    await nextTick();
    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    click(document.querySelector('.layer')!);
    await nextTick();
    expect(layerOpen()).toBe(false);
  });
});
