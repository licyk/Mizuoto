<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useDoctor, useEngineActions, useEngines } from '@/api/queries/engines';
import { useModelActions, useModels } from '@/api/queries/models';
import ErrorNotice from '@/components/ErrorNotice.vue';
import LicenceDialog from '@/components/LicenceDialog.vue';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, ProgressBar, icons } from '@/ui';

/**
 * What a model needs is offered, not reported: while its engine is disabled, its licence not
 * accepted, a package it needs not installed (torch, a licence-gated engine's own package) or the model
 * not downloaded, this replaces its slot (the Speak button) with the one action that fixes the first of
 * these, with progress while it runs; a missing package is shown with the command that installs it (the
 * server cannot install packages into itself). Once all is ready the slot shows again and ``ready`` is emitted.
 * ``compact`` offers it in the slot's place within a row (beside a field), without the gate's own panel.
 */
const props = defineProps<{ model: string | null; compact?: boolean }>();
const emit = defineEmits<{ ready: [] }>();
const { t, tx } = useI18n();
const models = useModels();
const engines = useEngines();
const modelActions = useModelActions();
const engineActions = useEngineActions();
const jobs = useJobsStore();
const licenceOpen = ref(false);
const error = ref<unknown>(null);

const info = computed(() => (models.data.value ?? []).find((m) => m.id === props.model) ?? null);
const engine = computed(() => (engines.data.value ?? []).find((e) => e.id === info.value?.engine) ?? null);
const runningJob = (kind: string, match: (request: Record<string, unknown>) => boolean) =>
  jobs.active.find((j) => j.kind === kind && match(j.request as Record<string, unknown>)) ?? null;
const downloadJob = computed(() => (info.value ? runningJob('model_download', (r) => ((r.models as string[]) ?? []).includes(info.value!.id)) : null));

type Need = 'enable' | 'licence' | 'missing' | 'download' | null;
const need = computed<Need>(() => {
  if (!info.value || !engine.value) return null;
  if (!engine.value.enabled) return 'enable';
  if (!engine.value.licence_accepted) return 'licence';
  if (!engine.value.available) return 'missing';
  if (!['installed', 'builtin'].includes(info.value.state)) return 'download';
  return null;
});
const busy = computed(() => need.value === 'download' && (!!downloadJob.value || info.value?.state === 'downloading'));
const job = computed(() => downloadJob.value);
// What installs the missing packages: torch's build for this machine (the doctor knows it), or the package itself.
const doctor = useDoctor(() => engine.value?.id ?? null, () => need.value === 'missing' && !engine.value?.remote);
const installHint = computed(() =>
  (engine.value?.missing ?? []).map((p) => (p === 'torch' ? (doctor.data.value?.install_command ?? 'mizuoto doctor') : `pip install ${p}`)).join(' && '),
);

watch(need, (n, before) => {
  if (n === null && before !== null && before !== undefined) emit('ready');
});

const onError = (e: unknown) => (error.value = e);
function act() {
  error.value = null;
  if (!engine.value || !info.value) return;
  if (need.value === 'enable') engineActions.configure.mutate({ engine: engine.value.id, patch: { enabled: true } }, { onError });
  else if (need.value === 'licence') licenceOpen.value = true;
  else if (need.value === 'download') modelActions.download.mutate([info.value.id], { onError });
}
const label = computed(() => {
  const e = tx(engine.value?.title);
  if (need.value === 'enable') return t('engines.enable');
  if (need.value === 'licence') return t('gate.licence', { engine: e });
  if (need.value === 'missing') return t('gate.missing', { engine: e, list: engine.value?.missing.join(', ') ?? '' });
  return t('gate.download', { name: info.value?.title ?? '', size: info.value?.size_mb ? formatBytes(info.value.size_mb * 2 ** 20) : '' });
});
</script>

<template>
  <slot v-if="!model || need === null" />
  <div v-else class="gate" :class="{ compact }">
    <ErrorNotice v-if="error" :error="error" />
    <div class="row">
      <ProgressBar v-if="busy" class="progress" :value="job?.progress ?? null" :label="job?.message || t('gate.busy')" />
      <span v-if="busy" class="type-body-small">{{ job?.message || t('gate.busy') }}</span>
      <div v-else-if="need === 'missing'" class="missing type-body-small">
        <span>{{ label }}</span>
        <span v-if="engine?.remote">{{ t('gate.missingRemote', { host: engine.remote }) }}</span>
        <code v-else>{{ installHint }}</code>
      </div>
      <AppButton v-else :variant="compact ? 'tonal' : 'filled'" :icon="need === 'download' ? icons.Download : need === 'licence' ? icons.ShieldCheck : icons.Package" @click="act">{{ label }}</AppButton>
    </div>
    <LicenceDialog v-model:open="licenceOpen" :engine="engine" @accept="(key) => engine && engineActions.acceptLicence.mutate({ engine: engine.id, licence: key }, { onError })" />
  </div>
</template>

<style scoped>
.gate { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3) var(--app-space-4); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-tertiary-container); color: var(--md-sys-color-on-tertiary-container); }
.row { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-3); }
.row > :first-child { flex: 1; min-width: 0; }
.missing { display: flex; flex-direction: column; gap: var(--app-space-1); }
.missing code { overflow-wrap: anywhere; user-select: all; }
.gate.compact { padding: 0; background: none; color: inherit; min-width: 0; }
.compact .row { justify-content: flex-start; gap: var(--app-space-2); }
.compact .row > :first-child { flex: 0 1 auto; }
.compact .row > .progress { flex: 0 0 120px; }
</style>
