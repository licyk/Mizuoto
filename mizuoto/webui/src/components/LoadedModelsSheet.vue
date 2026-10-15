<script setup lang="ts">
import { computed } from 'vue';
import { useReleaseCompute } from '@/api/queries/compute';
import { useRuntime, useUnloadModels } from '@/api/queries/runtime';
import { errorView } from '@/components/errors';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { AppButton, AppCard, Badge, EmptyState, IconButton, ProgressBar, SideSheet, TRANSITIONS, icons, staggerStyle, useSnackbar } from '@/ui';

/**
 * The models in memory, in the server's process (or on a remote host): each with its engine, device, the bytes of
 * weights it holds, how much of them is on its device (a model that does not fit streams the rest from RAM as it runs)
 * and Unload; Free GPU memory (idle engines unload their models). Engines whose last request failed say why.
 */
const open = defineModel<boolean>('open', { default: false });
const { t, tOr } = useI18n();
const runtime = useRuntime();
const unload = useUnloadModels();
const release = useReleaseCompute();
const snackbar = useSnackbar();
const models = computed(() => runtime.data.value?.models ?? []);
const failed = computed(() => (runtime.data.value?.engines ?? []).filter((e) => e.last_error));
const onError = (e: unknown) => snackbar.error(errorView(e, t, tOr).message);

/** Where a model's weights are: all on its device, part of them (the rest streams), none (moved to RAM), or ``null``
 * when there is nothing to show (the CPU, a remote host). */
function residency(m: { device: string; bytes?: number | null; loaded_bytes?: number | null; streamed_bytes?: number | null }) {
  if (m.device === 'cpu' || m.loaded_bytes == null || !m.bytes) return null;
  if (!m.loaded_bytes) return { share: 0, text: t('runtime.inRam') };
  if (m.streamed_bytes) return { share: m.loaded_bytes / m.bytes, text: t('runtime.partly', { size: formatBytes(m.loaded_bytes) }) };
  return { share: 1, text: t('runtime.whole') };
}

function free() {
  release.mutate(undefined, { onSuccess: () => snackbar.show(t('runtime.freed')), onError });
}
</script>

<template>
  <SideSheet v-model:open="open" :title="t('runtime.title')" :close-label="t('common.close')">
    <template #header-actions>
      <AppButton variant="text" :icon="icons.MemoryStick" :disabled="release.isPending.value" @click="free">{{ t('runtime.free') }}</AppButton>
    </template>
    <EmptyState v-if="!models.length && !failed.length" :icon="icons.Cpu" :title="t('runtime.none')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="list">
      <AppCard v-for="(m, i) in models" :key="`${m.model_id}@${m.device}`" variant="outlined" class="model" :style="staggerStyle(i)">
        <div class="head">
          <div class="titles">
            <span class="type-title-small">{{ m.model_id }}</span>
            <span class="type-body-small muted">
              {{ m.device }}<template v-if="m.bytes"> · {{ formatBytes(m.bytes) }}</template>
              <template v-if="m.remote"> · {{ t('remote.onHost', { host: m.remote }) }}</template>
            </span>
          </div>
          <Badge :value="m.busy ? t('runtime.busy') : t('runtime.idle')" :tone="m.busy ? 'primary' : 'neutral'" />
          <IconButton :icon="icons.CircleStop" :label="t('runtime.unload')" :disabled="m.busy || unload.isPending.value" @click="unload.mutate(m.model_id, { onError })" />
        </div>
        <template v-if="residency(m)">
          <ProgressBar :value="residency(m)?.share" :label="residency(m)?.text" />
          <span class="type-body-small muted">{{ residency(m)?.text }}</span>
        </template>
      </AppCard>
      <AppCard v-for="e in failed" :key="`${e.engine}:${e.device}:error`" variant="outlined" class="model">
        <div class="head">
          <div class="titles">
            <span class="type-title-small">{{ e.engine }} · {{ e.device }}</span>
          </div>
          <Badge :value="tOr(`runtime.states.${e.state}`, e.state)" tone="error" />
        </div>
        <p class="type-body-small err">{{ e.last_error?.message }}</p>
      </AppCard>
    </TransitionGroup>
  </SideSheet>
</template>

<style scoped>
.list { position: relative; display: flex; flex-direction: column; gap: var(--app-space-2); }
.model { display: flex; flex-direction: column; gap: var(--app-space-1); padding: var(--app-space-3); }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.titles { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.titles > span { overflow: hidden; text-overflow: ellipsis; }
.err { margin: 0; color: var(--md-sys-color-error); overflow-wrap: anywhere; }
</style>
