<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import { copyText, errorReport, errorView } from '@/components/errors';
import { useI18n } from '@/i18n';
import { AppIcon, IconButton, icons } from '@/ui';

/**
 * A domain error in place: its translated title (for a device, which one and why), the server's
 * message, and on request the detail field by field, tracebacks and process output preformatted,
 * with a button that copies all of it for a bug report.
 * ``dismissible`` adds a close button for an error that otherwise stays until it is resolved.
 */
const props = defineProps<{ error: unknown; dismissible?: boolean }>();
defineEmits<{ dismiss: [] }>();
const { t, tOr } = useI18n();
const view = computed(() => errorView(props.error, t, tOr));
const open = ref(false);
const hasDetail = computed(() => view.value.fields.length > 0 || view.value.blocks.length > 0);
const copied = ref(false);
let copiedTimer: ReturnType<typeof setTimeout> | undefined;

async function copy() {
  if (!(await copyText(errorReport(view.value, t('errors.fields.code'))))) return;
  copied.value = true;
  clearTimeout(copiedTimer);
  copiedTimer = setTimeout(() => (copied.value = false), 2000);
}

onBeforeUnmount(() => clearTimeout(copiedTimer));
</script>

<template>
  <div class="error-notice" role="alert">
    <AppIcon :icon="icons.AlertTriangle" :size="20" />
    <div class="body">
      <span class="type-title-small title">{{ view.title }}</span>
      <span v-if="view.message && view.message !== view.title" class="type-body-medium message">{{ view.message }}</span>
      <span v-if="view.hint" class="type-body-small hint">{{ view.hint }}</span>
      <div v-if="hasDetail" class="links">
        <button type="button" class="toggle type-label-large" :aria-expanded="open" @click="open = !open">{{ t('common.details') }}</button>
        <button v-if="open" type="button" class="toggle type-label-large" @click="copy">{{ copied ? t('errors.copied') : t('errors.copy') }}</button>
      </div>
      <div v-if="open" class="detail type-body-small">
        <dl v-if="view.fields.length" class="fields">
          <template v-for="f in view.fields" :key="f.key">
            <dt>{{ f.label }}</dt>
            <dd>{{ f.value }}</dd>
          </template>
        </dl>
        <section v-for="b in view.blocks" :key="b.key" class="block">
          <span class="block-label">{{ b.label }}</span>
          <pre>{{ b.value }}</pre>
        </section>
      </div>
    </div>
    <IconButton v-if="dismissible" class="dismiss" :icon="icons.X" :label="t('common.dismiss')" @click="$emit('dismiss')" />
  </div>
</template>

<style scoped>
.error-notice { display: flex; gap: var(--app-space-3); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-error-container); color: var(--md-sys-color-on-error-container); min-width: 0; }
.body { display: flex; flex-direction: column; gap: var(--app-space-1); min-width: 0; flex: 1; }
.hint { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; }
.message { overflow-wrap: anywhere; }
.links { display: flex; gap: var(--app-space-4); }
.toggle { align-self: flex-start; padding: 0; border: 0; background: none; color: inherit; text-decoration: underline; cursor: pointer; }
.dismiss { flex: none; align-self: flex-start; margin: calc(-1 * var(--app-space-2)) calc(-1 * var(--app-space-2)) 0 0; --md-icon-button-icon-color: currentColor; --md-icon-button-hover-icon-color: currentColor; --md-icon-button-focus-icon-color: currentColor; --md-icon-button-pressed-icon-color: currentColor; }
.detail { display: flex; flex-direction: column; gap: var(--app-space-2); margin-top: var(--app-space-1); min-width: 0; }
.fields { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: var(--app-space-1) var(--app-space-3); margin: 0; }
.fields dt { opacity: 0.8; }
.fields dd { margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.block { display: flex; flex-direction: column; gap: var(--app-space-1); min-width: 0; }
.block-label { opacity: 0.8; }
.block pre { margin: 0; padding: var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); background: color-mix(in srgb, currentColor 8%, transparent); white-space: pre; overflow: auto; max-height: 240px; font-family: ui-monospace, monospace; }
</style>
