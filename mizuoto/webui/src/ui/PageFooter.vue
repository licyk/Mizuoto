<script setup lang="ts">
/**
 * A page's action bar, held at the bottom of the scroller under the page's one column of sections:
 * the run button, the jobs it started and their errors stay in view however far the page scrolls.
 */
</script>

<template>
  <!-- data-page-footer: in-page dropdowns scroll clear of it (ui/reveal.ts). -->
  <footer class="page-footer" data-page-footer><slot /></footer>
</template>

<style scoped>
.page-footer {
  position: sticky; bottom: 0; z-index: var(--app-z-page-footer); display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3) var(--app-space-6);
  background: var(--md-sys-color-surface-container); border-top: 1px solid var(--md-sys-color-outline-variant);
}
@container app-content (max-width: 599px) {
  .page-footer { padding: var(--app-space-3); }
}
</style>
