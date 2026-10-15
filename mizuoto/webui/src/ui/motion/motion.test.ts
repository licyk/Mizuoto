import { mount } from '@vue/test-utils';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { defineComponent, h, nextTick, ref, Transition } from 'vue';
import { prefersReducedMotion, TRANSITIONS } from '@/ui/motion/transitions';

const SRC = join(process.cwd(), 'src');

function useStylesheet(path: string) {
  const style = document.createElement('style');
  style.textContent = readFileSync(join(SRC, path), 'utf8');
  document.head.appendChild(style);
  return () => style.remove();
}

/** A leaving element must be removed even when the transition never reports an end (zero duration). */
async function smoke() {
  const shown = ref(true);
  const Comp = defineComponent(() => () => h(Transition, { name: TRANSITIONS.fadeThrough }, () => (shown.value ? h('div', { class: 'box' }, 'x') : null)));
  const wrapper = mount(Comp, { attachTo: document.body });
  expect(wrapper.find('.box').exists()).toBe(true);
  shown.value = false;
  await nextTick();
  await new Promise((r) => setTimeout(r, 30));
  expect(wrapper.find('.box').exists()).toBe(false);
  shown.value = true;
  await nextTick();
  expect(wrapper.find('.box').exists()).toBe(true);
  wrapper.unmount();
}

describe('motion', () => {
  let cleanup: (() => void)[] = [];
  afterEach(() => {
    cleanup.forEach((f) => f());
    cleanup = [];
    delete document.documentElement.dataset.motion;
  });

  it('runs with full motion under the zero-duration test stylesheet', async () => {
    cleanup.push(useStylesheet('ui/motion/motion.css'), useStylesheet('test/motion-zero.css'));
    expect(prefersReducedMotion()).toBe(false);
    await smoke();
  });

  it('runs with reduced motion, from the in-app choice', async () => {
    cleanup.push(useStylesheet('ui/motion/motion.css'), useStylesheet('test/motion-zero.css'));
    document.documentElement.dataset.motion = 'reduced';
    expect(prefersReducedMotion()).toBe(true);
    await smoke();
  });

  it('defines the reduced-motion rules for the system setting and for the in-app choice', () => {
    const css = readFileSync(join(SRC, 'ui/motion/motion.css'), 'utf8');
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toContain(":root[data-motion='reduced'] [class*='-enter-active']");
    for (const name of Object.values(TRANSITIONS)) expect(css).toContain(`.${name}-enter`);
  });

  it('wraps destination changes in a view transition where the browser has one', async () => {
    const { createRouter, createMemoryHistory } = await import('vue-router');
    const { installViewTransitions } = await import('@/ui/motion/viewTransition');
    const calls: string[] = [];
    (document as unknown as { startViewTransition: unknown }).startViewTransition = (cb: () => Promise<void>) => {
      calls.push('start');
      // As browsers do: the update callback runs after the old state is captured (a task later).
      const updateCallbackDone = new Promise<void>((r) => setTimeout(r, 0)).then(cb).then(() => {
        calls.push('done');
      });
      return { finished: updateCallbackDone, ready: updateCallbackDone, updateCallbackDone };
    };
    const C = defineComponent(() => () => h('div'));
    const router = createRouter({ history: createMemoryHistory(), routes: [{ path: '/a', name: 'a', component: C }, { path: '/b', name: 'b', component: C }] });
    const supported = installViewTransitions(router);
    expect(supported.value).toBe(true);
    await router.push('/a');
    await router.push('/b');
    await new Promise((r) => setTimeout(r, 50));
    expect(calls).toEqual(['start', 'done']);
    // Reduced motion still changes destinations through a (short, CSS-defined) view transition.
    document.documentElement.dataset.motion = 'reduced';
    await router.push('/a');
    await new Promise((r) => setTimeout(r, 50));
    expect(calls).toEqual(['start', 'done', 'start', 'done']);
    delete document.documentElement.dataset.motion;
    delete (document as unknown as { startViewTransition?: unknown }).startViewTransition;
  });

  it('morphs the pair it names both ways, and drops the names when the transition ends', async () => {
    const { createRouter, createMemoryHistory } = await import('vue-router');
    const { installViewTransitions, morphStyle } = await import('@/ui/motion/viewTransition');
    const seen: Record<string, string>[] = [];
    (document as unknown as { startViewTransition: unknown }).startViewTransition = (cb: () => Promise<void>) => {
      seen.push(morphStyle('card:1'));
      const updateCallbackDone = new Promise<void>((r) => setTimeout(r, 0)).then(cb);
      return { finished: updateCallbackDone, ready: updateCallbackDone, updateCallbackDone };
    };
    const C = defineComponent(() => () => h('div'));
    const routes = [{ path: '/list', name: 'list', component: C }, { path: '/item/:id', name: 'item', component: C }, { path: '/other', name: 'other', component: C }];
    const router = createRouter({ history: createMemoryHistory(), routes });
    installViewTransitions(router, {
      morph: (to, from) => (to.name === 'item' && from.name === 'list' ? `card:${String(to.params.id)}` : from.name === 'item' && to.name === 'list' ? `card:${String(from.params.id)}` : null),
    });
    await router.push('/list');
    for (const path of ['/item/1', '/list', '/other']) {
      await router.push(path);
      await new Promise((r) => setTimeout(r, 20));
    }
    const morph = { viewTransitionName: 'morph', viewTransitionGroup: 'nearest' };
    expect(seen).toEqual([morph, morph, {}]);
    expect(morphStyle('card:1')).toEqual({});
    // Reduced motion names no pair: the page's crossfade alone.
    document.documentElement.dataset.motion = 'reduced';
    await router.push('/list');
    await router.push('/item/1');
    await new Promise((r) => setTimeout(r, 20));
    expect(seen.at(-1)).toEqual({});
    delete document.documentElement.dataset.motion;
    delete (document as unknown as { startViewTransition?: unknown }).startViewTransition;
  });

  it('gives reduced motion a short crossfade between destinations', () => {
    const css = readFileSync(join(SRC, 'ui/motion/motion.css'), 'utf8');
    expect(css).toContain(":root[data-motion='reduced']::view-transition-new(app-content)");
  });
});
