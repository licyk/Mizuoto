<script setup lang="ts">
import { nextTick, ref, watch } from 'vue';
import IconButton from '@/ui/IconButton.vue';
import { X } from '@/ui/icons';
import { containerFrom } from '@/ui/motion/transitions';
import { useLayer } from '@/ui/layers';

/**
 * A modal dialog built from the tokens. With ``fromRect`` it grows from that rectangle (the
 * ``container`` transition); otherwise it fades and scales.
 */
const props = withDefaults(defineProps<{ title?: string; fromRect?: DOMRect | null; width?: 'small' | 'medium' | 'large'; closeLabel?: string }>(), { width: 'medium', closeLabel: 'Close' });
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ closed: [] }>();
const panel = ref<HTMLElement | null>(null);
const motionStyle = ref<Record<string, string>>({});
// Escape, a click outside, focus in and back out, and the page made inert are the layer's.
const { zIndex } = useLayer({ kind: 'dialog', modal: true, open: () => open.value, close: () => (open.value = false), elements: () => [panel.value], initialFocus: () => panel.value });

watch(
  open,
  async (value) => {
    if (!value) return;
    motionStyle.value = {};
    await nextTick();
    // Offsets ignore the enter transform already applied, unlike getBoundingClientRect().
    const p = panel.value;
    motionStyle.value = p ? containerFrom(props.fromRect, new DOMRect(p.offsetLeft, p.offsetTop, p.offsetWidth, p.offsetHeight)) : {};
  },
  { immediate: true },
);
</script>

<template>
  <Teleport to="body">
    <Transition name="scrim">
      <div v-if="open" class="scrim" :style="{ zIndex }" />
    </Transition>
    <Transition name="container" @after-leave="emit('closed')">
      <div v-if="open" class="layer" :style="{ zIndex }">
        <section ref="panel" class="dialog" :class="width" role="dialog" aria-modal="true" :aria-label="title" tabindex="-1" :style="motionStyle">
          <header v-if="title || $slots.header" class="head">
            <slot name="header"><h2 class="type-headline-small title">{{ title }}</h2></slot>
            <IconButton :icon="X" :label="closeLabel" @click="open = false" />
          </header>
          <div class="body"><slot /></div>
          <footer v-if="$slots.actions" class="actions"><slot name="actions" /></footer>
        </section>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.scrim { position: fixed; inset: 0; background: color-mix(in srgb, var(--md-sys-color-scrim) 32%, transparent); }
/* The explicit minmax(0, …) column is what keeps a panel with wide content — a long file name, a
   metadata table — inside the window instead of letting the track grow to its minimum size. */
.layer { position: fixed; inset: 0; display: grid; grid-template-columns: minmax(0, 1fr); place-items: center; padding: var(--app-space-4); }
.dialog {
  display: flex; flex-direction: column; max-height: calc(100vh - 32px); width: 100%; min-width: 0; outline: none;
  background: var(--md-sys-color-surface-container-high); color: var(--md-sys-color-on-surface);
  border-radius: var(--md-sys-shape-corner-extra-large); box-shadow: var(--app-elevation-3);
}
.small { max-width: var(--app-width-dialog-small); }
.medium { max-width: var(--app-width-dialog-medium); }
.large { max-width: var(--app-width-dialog-large); }
.head { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-2); padding: var(--app-space-4) var(--app-space-4) 0 var(--app-space-6); }
/* A model name can be one long unbroken word: break it, and keep the header at two lines. */
.title { margin: 0; overflow: hidden; text-overflow: ellipsis; overflow-wrap: anywhere; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; }
.body { padding: var(--app-space-4) var(--app-space-6); overflow: auto; flex: 1; }
.actions { display: flex; justify-content: flex-end; flex-wrap: wrap; gap: var(--app-space-2); padding: 0 var(--app-space-6) var(--app-space-6); }
@media (max-width: 599px) {
  .layer { padding: 0; align-items: end; }
  /* A bottom sheet on a phone: dvh follows the browser's collapsing toolbars, and the inset keeps
     the last row clear of the home indicator. */
  .dialog { max-height: 92vh; max-height: 92dvh; padding-bottom: env(safe-area-inset-bottom); border-radius: var(--md-sys-shape-corner-extra-large) var(--md-sys-shape-corner-extra-large) 0 0; }
  .title { font-size: var(--md-sys-typescale-title-large-size); line-height: var(--md-sys-typescale-title-large-line-height); }
  .head { padding: var(--app-space-4) var(--app-space-2) 0 var(--app-space-4); }
  .body { padding: var(--app-space-3) var(--app-space-4); }
  .actions { padding: 0 var(--app-space-4) var(--app-space-4); }
}
</style>
