<script setup lang="ts">
/**
 * Label/value rows. Values may be long (a prompt, a hash) and wrap; below 600 px the label goes
 * above its value. Rows come from ``rows`` or from the default slot as <dt>/<dd> pairs.
 */
defineProps<{ rows?: { label: string; value: string | number | null | undefined; mono?: boolean }[] }>();
</script>

<template>
  <dl class="data-list type-body-medium">
    <template v-for="(row, i) in rows ?? []" :key="i">
      <template v-if="row.value !== null && row.value !== undefined && row.value !== ''">
        <dt class="muted">{{ row.label }}</dt>
        <dd :class="{ mono: row.mono }">{{ row.value }}</dd>
      </template>
    </template>
    <slot />
  </dl>
</template>

<style scoped>
.data-list { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: var(--app-space-1) var(--app-space-4); margin: 0; }
.data-list :deep(dt) { white-space: nowrap; }
.data-list :deep(dd) { margin: 0; min-width: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: var(--md-sys-typescale-body-small-size); }
@media (max-width: 599px) {
  .data-list { grid-template-columns: minmax(0, 1fr); gap: 0; }
  .data-list :deep(dd) { margin-bottom: var(--app-space-2); }
}
</style>
