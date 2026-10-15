<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { fetchJobLog } from '@/api/queries/jobs';
import { onJobLog } from '@/api/socket';
import { useI18n } from '@/i18n';

/** A job's log: fetched over REST, extended by ``job_log`` events. Tracebacks live here, not in the UI. */
const props = defineProps<{ jobId: string }>();
const { t } = useI18n();
const lines = ref<string[]>([]);
const offset = ref(0);
const box = ref<HTMLElement | null>(null);
let stop: (() => void) | null = null;

async function more() {
  const chunk = await fetchJobLog(props.jobId, offset.value);
  lines.value.push(...chunk.lines);
  offset.value = chunk.next_offset;
  await nextTick();
  if (box.value) box.value.scrollTop = box.value.scrollHeight;
}

onMounted(async () => {
  await more().catch(() => undefined);
  stop = onJobLog(props.jobId, (eventOffset) => {
    // Events carry at most 50 lines; fetch from where we are, so nothing is missed or doubled.
    if (eventOffset >= offset.value) more().catch(() => undefined);
  });
});
onBeforeUnmount(() => stop?.());
</script>

<template>
  <pre ref="box" class="log type-body-small">{{ lines.length ? lines.join('\n') : t('jobs.noLog') }}</pre>
</template>

<style scoped>
.log { margin: 0; max-height: 240px; overflow: auto; padding: var(--app-space-2) var(--app-space-3); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container-highest); white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, monospace; }
</style>
