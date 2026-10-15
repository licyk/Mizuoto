<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import { useLayer } from '@/ui/layers';
import type { MenuItem } from '@/ui/AppMenu.vue';

/**
 * A menu at a point: a right-click or a long press on a grid cell. Rendered at the end of the
 * document, kept inside the window, closed by Escape, a click elsewhere or a scroll of what is
 * behind it (scrolling the menu's own long list keeps it open).
 * An item with ``divider`` draws a line above itself.
 */
const props = defineProps<{ items: (MenuItem & { divider?: boolean })[]; x: number; y: number }>();
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ select: [string] }>();
const list = ref<HTMLElement | null>(null);
const position = ref<Record<string, string>>({});
const MARGIN = 8;
// Escape or a click elsewhere closes the menu alone, not the dialog it opened from.
const layer = useLayer({
  kind: 'menu',
  open: () => open.value,
  close: () => (open.value = false),
  elements: () => [list.value],
  initialFocus: () => list.value?.querySelector<HTMLButtonElement>('button:not([disabled])'),
});

function place() {
  const menu = list.value?.getBoundingClientRect();
  const width = menu?.width ?? 220;
  const height = menu?.height ?? 0;
  const left = Math.min(props.x, window.innerWidth - width - MARGIN);
  const top = props.y + height > window.innerHeight - MARGIN ? Math.max(MARGIN, props.y - height) : props.y;
  position.value = { left: `${Math.max(MARGIN, left)}px`, top: `${top}px`, maxHeight: `${window.innerHeight - MARGIN * 2}px` };
}

const onKey = (e: KeyboardEvent) => {
  if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && layer.isTop()) {
    const items = [...(list.value?.querySelectorAll<HTMLButtonElement>('button:not([disabled])') ?? [])];
    const i = items.indexOf(document.activeElement as HTMLButtonElement);
    items[(i + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus();
    e.preventDefault();
  }
};
const close = () => (open.value = false);
// Scroll events are listened to in the capture phase, so the menu's own list scrolling arrives here too.
const onScroll = (e: Event) => {
  if (!list.value?.contains(e.target as Node)) close();
};

function listen(on: boolean) {
  if (on) {
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', close);
    window.addEventListener('blur', close);
  } else {
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('scroll', onScroll, true);
    window.removeEventListener('resize', close);
    window.removeEventListener('blur', close);
  }
}

watch(open, async (v) => {
  listen(v);
  if (v) {
    await nextTick();
    place();
  }
});
watch(() => [props.x, props.y], () => open.value && nextTick(place));
onBeforeUnmount(() => listen(false));

function choose(id: string) {
  open.value = false;
  emit('select', id);
}
</script>

<template>
  <Teleport to="body">
    <Transition name="snackbar">
      <div v-if="open" ref="list" class="menu" role="menu" :style="{ ...position, zIndex: layer.zIndex.value }" @contextmenu.prevent>
        <template v-for="item in items" :key="item.id">
          <div v-if="item.divider" class="divider" role="separator" />
          <button type="button" role="menuitem" class="item state-layer type-label-large" :class="{ danger: item.danger }" :disabled="item.disabled" @click="choose(item.id)">
            <AppIcon v-if="item.icon" :icon="item.icon" :size="20" />
            <span class="label">{{ item.label }}</span>
          </button>
        </template>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.menu {
  position: fixed; min-width: 220px; max-width: min(var(--app-width-menu), calc(100vw - 16px)); padding: var(--app-space-2) 0; overflow-y: auto;
  background: var(--md-sys-color-surface-container); border-radius: var(--md-sys-shape-corner-extra-small); box-shadow: var(--app-elevation-2);
}
.item {
  display: flex; align-items: center; gap: var(--app-space-3); width: 100%; height: 40px; padding: 0 var(--app-space-3);
  border: 0; background: transparent; color: var(--md-sys-color-on-surface); cursor: pointer; text-align: left; font: inherit;
}
.label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.item:disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.item.danger { color: var(--md-sys-color-error); }
.divider { height: 1px; margin: var(--app-space-1) 0; background: var(--md-sys-color-outline-variant); }
</style>
