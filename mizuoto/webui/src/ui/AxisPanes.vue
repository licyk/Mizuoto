<script setup lang="ts" generic="T extends string">
import { ref, toRef } from 'vue';
import { TRANSITIONS, useAxisDirection } from '@/ui/motion/transitions';

/**
 * Panes switched by a segmented control or tabs: the default slot renders one keyed pane (v-if /
 * v-else on ``value``), or none. A change slides the way the control moved (shared-axis-x), and the
 * box eases to the new pane's height over the same time, so what follows glides rather than jumps.
 * The box clips only during a change: a taller leaving pane would draw over what follows, and a
 * pane's picker may list in place.
 */
const props = defineProps<{ value: T; order: readonly T[] }>();
const dir = useAxisDirection(toRef(props, 'value'), () => props.order);
const box = ref<HTMLElement | null>(null);
let changing = false;
let entering: HTMLElement | null = null;

function release() {
  const el = box.value;
  changing = false;
  if (el) el.style.height = el.style.transition = el.style.overflow = '';
}

// Leave and enter hooks of one change run in the same tick; the first holds the old height, and a
// frame later the box eases to the entering pane's height (0 when none enters).
function begin() {
  const el = box.value;
  if (!el || changing) return;
  changing = true;
  entering = null;
  el.style.height = `${el.offsetHeight}px`;
  el.style.overflow = 'clip';
  requestAnimationFrame(() => {
    const target = entering ? entering.offsetHeight : 0;
    if (target === el.offsetHeight) return release();
    el.style.transition = 'height var(--md-sys-motion-duration-medium2) var(--md-sys-motion-easing-emphasized)';
    el.style.height = `${target}px`;
    if (parseFloat(getComputedStyle(el).transitionDuration) === 0) release();
  });
}
function onEnter(pane: Element) {
  entering = pane as HTMLElement;
}
function onTransitionEnd(e: TransitionEvent) {
  if (e.target === box.value && e.propertyName === 'height') release();
}
</script>

<template>
  <div ref="box" class="axis-panes" :style="{ '--axis-dir': dir }" @transitionend="onTransitionEnd">
    <Transition :name="TRANSITIONS.sharedAxisX" @before-leave="begin" @before-enter="begin" @enter="onEnter">
      <slot />
    </Transition>
  </div>
</template>

<style scoped>
/* The leaving pane is taken out of the flow (shared-axis-x) and slides inside this box. */
.axis-panes { position: relative; overflow-x: clip; min-width: 0; }
</style>
