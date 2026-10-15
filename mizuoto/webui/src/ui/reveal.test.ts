import { describe, expect, it, vi } from 'vitest';
import { revealDropdown } from '@/ui/reveal';

function box(top: number, bottom: number): DOMRect {
  return { top, bottom, left: 0, right: 100, width: 100, height: bottom - top, x: 0, y: top, toJSON: () => ({}) } as DOMRect;
}

/** A scroller (0–800) holding a dropdown and, optionally, a sticky footer from 700. */
function page(list: [number, number], footer: boolean) {
  const scroller = document.createElement('main');
  scroller.style.overflowY = 'auto';
  scroller.getBoundingClientRect = () => box(0, 800);
  scroller.scrollBy = vi.fn() as typeof scroller.scrollBy;
  const el = document.createElement('div');
  el.getBoundingClientRect = () => box(...list);
  scroller.append(el);
  if (footer) {
    const f = document.createElement('footer');
    f.dataset.pageFooter = '';
    f.getBoundingClientRect = () => box(700, 800);
    scroller.append(f);
  }
  document.body.append(scroller);
  return { scroller, el };
}

describe('revealDropdown', () => {
  it('scrolls a list that reaches under the page footer up until it clears it', () => {
    const { scroller, el } = page([500, 760], true);
    revealDropdown(el);
    expect(scroller.scrollBy).toHaveBeenCalledWith(expect.objectContaining({ top: 68 }));
  });

  it('leaves a list that already fits alone', () => {
    const { scroller, el } = page([300, 600], true);
    revealDropdown(el);
    expect(scroller.scrollBy).not.toHaveBeenCalled();
  });

  it('never scrolls past the list top', () => {
    const { scroller, el } = page([40, 900], false);
    revealDropdown(el);
    expect(scroller.scrollBy).toHaveBeenCalledWith(expect.objectContaining({ top: 32 }));
  });
});

describe('placeDropdown', () => {
  /** A page (0–1000) whose sticky footer is 100 high, holding a field at ``top``–``top + 50``. */
  function field(top: number) {
    const scroller = document.createElement('main');
    scroller.style.overflowY = 'auto';
    const pageEl = document.createElement('div');
    pageEl.getBoundingClientRect = () => box(0, 1000);
    const anchor = document.createElement('button');
    anchor.getBoundingClientRect = () => box(top, top + 50);
    const f = document.createElement('footer');
    f.dataset.pageFooter = '';
    Object.defineProperty(f, 'offsetHeight', { value: 100 });
    pageEl.append(anchor, f);
    scroller.append(pageEl);
    document.body.append(scroller);
    return anchor;
  }

  it('opens under the field when the page has the room', async () => {
    const { placeDropdown } = await import('@/ui/reveal');
    expect(placeDropdown(field(100), 300)).toEqual({ above: false, reserve: 0 });
  });

  it('opens over a field near the page end rather than hang past it (the footer would rise)', async () => {
    const { placeDropdown } = await import('@/ui/reveal');
    expect(placeDropdown(field(700), 300)).toEqual({ above: true, reserve: 0 });
  });

  it('makes the room a list fits nowhere else', async () => {
    const { placeDropdown } = await import('@/ui/reveal');
    // 742 under the field (1000 - 100 - 150 - 8), 92 over it: 958 is 216 short.
    expect(placeDropdown(field(100), 958)).toEqual({ above: false, reserve: 216 });
  });
});

describe('revealDropdown over its field', () => {
  it('scrolls a list cut off at the top back into view', () => {
    const scroller = document.createElement('main');
    scroller.style.overflowY = 'auto';
    scroller.getBoundingClientRect = () => box(0, 800);
    scroller.scrollBy = vi.fn() as typeof scroller.scrollBy;
    const el = document.createElement('div');
    el.getBoundingClientRect = () => box(-120, 300);
    scroller.append(el);
    document.body.append(scroller);
    revealDropdown(el);
    expect(scroller.scrollBy).toHaveBeenCalledWith(expect.objectContaining({ top: -128 }));
  });
});
