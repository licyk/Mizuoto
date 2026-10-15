<script setup lang="ts">
/**
 * A field and what acts on it (a button, a toggle, icon buttons) side by side. The field takes the
 * room (``grow``) and the ``end`` slot sits centred on the field's own box, not on the box and the
 * supporting text under it, so a hint never pulls the button out of line. Wraps under the field
 * when the row is narrow.
 */
withDefaults(defineProps<{ grow?: boolean }>(), { grow: true });
</script>

<template>
  <div class="field-row">
    <div class="field" :class="{ grow }"><slot /></div>
    <div v-if="$slots.end" class="end"><slot name="end" /></div>
  </div>
</template>

<style scoped>
.field-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--app-space-2); min-width: 0; }
.field { display: flex; flex-direction: column; min-width: 0; }
.field.grow { flex: 1 1 240px; }
/* The height of an outlined field's box (md-outlined-text-field, md-outlined-select). */
.end { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); min-height: var(--app-field-height); }
</style>
