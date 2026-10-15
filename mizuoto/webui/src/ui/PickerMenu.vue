<script setup lang="ts">
import { computed, ref, watch, type Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';
import Badge from '@/ui/Badge.vue';
import ProgressCircle from '@/ui/ProgressCircle.vue';
import TextField from '@/ui/TextField.vue';
import { ChevronDown, Search } from '@/ui/icons';
import { useRevealWhileOpen } from '@/ui/reveal';
import { useLayer } from '@/ui/layers';

export interface PickerOption {
  value: string;
  label: string;
  /** A second line under the name: a voice's rate and version, a device's driver and channels. */
  description?: string;
  /** More words the search matches (a voice's tags). */
  keywords?: string[];
  /** Short tags after the name: "default", "virtual", "no index", "not connected". */
  badges?: { text: string; tone?: 'primary' | 'warning' | 'neutral' | 'error' }[];
  /** Listed, greyed and not selectable (a saved device that is not connected). */
  unavailable?: boolean;
}

/**
 * The one picker of the app (a voice, an audio device): a field showing the choice with its second
 * line and badges, opening a list in place, searchable when it is long. Names wrap rather than
 * being cut, in the field and in the list.
 */
const props = withDefaults(
  defineProps<{
    label: string;
    options: PickerOption[];
    icon?: Component;
    placeholder?: string;
    disabled?: boolean;
    /** The options are still coming: the field shows a spinner and the placeholder, and does not open. */
    loading?: boolean;
    /** The search field's label; without it there is no search. */
    searchLabel?: string;
    /** The search shows from this many options. */
    searchFrom?: number;
    /** Shown when the search matches nothing. */
    noMatches?: string;
  }>(),
  { searchFrom: 0, noMatches: '' },
);
const model = defineModel<string | null>({ default: null });
const open = ref(false);
const query = ref('');
const root = ref<HTMLElement | null>(null);
const popover = ref<HTMLElement | null>(null);
// Lists in place (its z-index is the page's), but Escape and a click elsewhere close it before the dialog it sits in.
useLayer({ kind: 'inline', open: () => open.value, close: () => (open.value = false), elements: () => [root.value] });
const field = ref<HTMLElement | null>(null);
const { above } = useRevealWhileOpen(() => open.value, popover, field);

const current = computed(() => props.options.find((o) => o.value === model.value) ?? null);
const searchable = computed(() => !!props.searchLabel && props.options.length >= props.searchFrom);
const shown = computed(() => {
  const q = query.value.trim().toLowerCase();
  if (!q || !searchable.value) return props.options;
  return props.options.filter((o) => [o.label, o.description ?? '', ...(o.keywords ?? [])].some((s) => s.toLowerCase().includes(q)));
});
watch(open, (isOpen) => {
  if (!isOpen) query.value = '';
});

function toggle() {
  if (!props.loading) open.value = !open.value;
}
function pick(o: PickerOption) {
  if (o.unavailable) return;
  model.value = o.value;
  open.value = false;
}
function onBlur(e: FocusEvent) {
  if (!root.value?.contains(e.relatedTarget as Node | null)) open.value = false;
}
</script>

<template>
  <div ref="root" class="picker" @focusout="onBlur">
    <button ref="field" type="button" class="field state-layer" :disabled="disabled" :aria-busy="loading" :aria-expanded="open" aria-haspopup="listbox" @click="toggle">
      <AppIcon v-if="icon" :icon="icon" :size="24" class="lead" />
      <span class="text">
        <span class="type-body-small muted">{{ label }}</span>
        <span class="type-title-medium name" :class="{ muted: loading || !current }">{{ loading ? (placeholder ?? '') : (current?.label ?? placeholder ?? '') }}</span>
        <span v-if="!loading && current?.description" class="type-body-small muted meta">{{ current.description }}</span>
      </span>
      <template v-if="!loading">
        <Badge v-for="b in current?.badges ?? []" :key="b.text" :value="b.text" :tone="b.tone ?? 'neutral'" />
      </template>
      <ProgressCircle v-if="loading" :size="20" :label="placeholder" class="trail" />
      <AppIcon v-else :icon="ChevronDown" :size="20" class="trail chevron" :class="{ open }" />
    </button>
    <Transition name="scrim">
      <div v-if="open" ref="popover" class="popover" :class="{ above }" role="dialog" :aria-label="label">
        <TextField v-if="searchable" v-model="query" :label="searchLabel" :icon="Search" type="search" />
        <ul class="list" role="listbox" :aria-label="label">
          <li v-for="o in shown" :key="o.value">
            <button type="button" role="option" class="option state-layer" :class="{ unavailable: o.unavailable }" :aria-selected="o.value === model" :aria-disabled="o.unavailable" @click="pick(o)">
              <span class="text">
                <span class="type-body-large name">{{ o.label }}</span>
                <span v-if="o.description" class="type-body-small muted meta">{{ o.description }}</span>
              </span>
              <Badge v-for="b in o.badges ?? []" :key="b.text" :value="b.text" :tone="b.tone ?? 'neutral'" />
            </button>
          </li>
        </ul>
        <p v-if="!shown.length && noMatches" class="type-body-medium muted empty">{{ noMatches }}</p>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.picker { position: relative; min-width: 0; }
.field { display: flex; align-items: center; gap: var(--app-space-3); width: 100%; min-width: 0; padding: var(--app-space-3) var(--app-space-4); border: 1px solid var(--md-sys-color-outline-variant); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer; }
.field:disabled { opacity: var(--app-disabled-content-opacity); cursor: default; }
.field[aria-busy='true'] { cursor: progress; }
.lead, .trail { flex: none; }
.text { display: flex; flex-direction: column; flex: 1; min-width: 0; }
/* Device and voice names are long (a driver's full endpoint name, an experiment with its epoch): wrap, never cut. */
.name, .meta { overflow-wrap: anywhere; }
.popover { position: absolute; z-index: var(--app-z-dropdown); top: calc(100% + 4px); left: 0; right: 0; display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3); background: var(--md-sys-color-surface-container); border-radius: var(--md-sys-shape-corner-medium); box-shadow: var(--app-elevation-2); }
.popover.above { top: auto; bottom: calc(100% + 4px); }
.list { margin: 0; padding: 0; list-style: none; max-height: 360px; overflow: auto; }
.option { display: flex; align-items: center; gap: var(--app-space-2); width: 100%; padding: var(--app-space-2) var(--app-space-3); border: 0; border-radius: var(--md-sys-shape-corner-small); background: transparent; color: var(--md-sys-color-on-surface); font: inherit; text-align: start; cursor: pointer; min-width: 0; }
.option[aria-selected='true'] { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.option.unavailable { opacity: var(--app-disabled-content-opacity); cursor: default; }
.empty { margin: 0; padding: var(--app-space-2) var(--app-space-3); }
.chevron { transition: transform var(--md-sys-motion-duration-short4) var(--md-sys-motion-easing-standard); }
.chevron.open { transform: rotate(180deg); }
</style>
