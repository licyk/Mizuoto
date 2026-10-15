<script setup lang="ts">
import { computed, ref } from 'vue';
import { useComputeInfo } from '@/api/queries/compute';
import { useDoctor, useEngineActions } from '@/api/queries/engines';
import type { EngineStatus } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import HyperparamsForm from '@/components/HyperparamsForm.vue';
import LicenceDialog from '@/components/LicenceDialog.vue';
import { useRemoteHosts } from '@/api/queries/remote';
import { useI18n } from '@/i18n';
import { AppButton, AppCard, Badge, ExpansionPanel, SelectField, Switch, icons } from '@/ui';

/**
 * One engine: what it is, its licence (accepted or to accept), whether it is available (every engine runs in the
 * server's process; one lacks torch, or a licence-gated engine its own package), its device, precision and load
 * options, and what the doctor finds for it.
 */
const props = defineProps<{ engine: EngineStatus; settings: Record<string, unknown> | null }>();
const { t, tx } = useI18n();
const actions = useEngineActions();
const compute = useComputeInfo();
const licenceOpen = ref(false);
const doctorOpen = ref(false);
const doctor = useDoctor(() => props.engine.id, doctorOpen);
const findings = computed(() => (doctor.data.value?.findings ?? []).filter((f) => f.check !== 'devices' && (!f.engine || f.engine === props.engine.id)));
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

const gated = computed(() => props.engine.licence.gated);
// This machine's devices, then each online remote host's (where the engine is available there).
// `cuda` is torch's device type, so it covers AMD GPUs under ROCm too (as ComputeService.resolve_device does).
const runsOn = (kind: string) => props.engine.devices.includes(kind) || (kind === 'rocm' && props.engine.devices.includes('cuda'));
const remoteHosts = useRemoteHosts();
const deviceOptions = computed(() => [
  { value: 'auto', label: t('common.auto') },
  ...(compute.data.value?.devices ?? []).filter((d) => runsOn(d.kind)).map((d) => ({ value: d.id, label: `${d.id} · ${d.name}` })),
  ...(remoteHosts.data.value ?? [])
    .filter((h) => h.info?.engines.some((e) => e.id === props.engine.id && e.ready) || h.engines.includes(props.engine.id))
    .flatMap((h) => [
      { value: `${h.name}/auto`, label: t('remote.device', { host: h.name, device: t('common.auto') }) },
      ...(h.info?.devices ?? []).filter((d) => runsOn(d.kind)).map((d) => ({ value: `${h.name}/${d.id}`, label: t('remote.device', { host: h.name, device: `${d.id} · ${d.name}` }) })),
    ]),
  // The saved device stays listed while its host is offline.
  ...(String(props.settings?.device ?? '').includes('/') && !(remoteHosts.data.value ?? []).some((h) => h.info && String(props.settings?.device).startsWith(`${h.name}/`))
    ? [{ value: String(props.settings?.device), label: String(props.settings?.device) }]
    : []),
]);
const device = computed(() => String(props.settings?.device ?? 'auto'));
const dtypeOptions = ['auto', 'bf16', 'fp16', 'fp32'].map((v) => ({ value: v, label: v === 'auto' ? t('common.auto') : v }));
const dtype = computed(() => String(props.settings?.dtype ?? 'auto'));
// How its models use GPU memory: Settings › Compute's mode, or one of its own.
const vramModeOptions = computed(() => [
  { value: 'auto', label: t('settings.engineVramModeInherit') },
  ...['normal', 'high', 'low', 'none'].map((v) => ({ value: v, label: t(`settings.vramModes.${v}`) })),
]);
const vramMode = computed(() => String(props.settings?.vram_mode ?? 'auto'));
// Its attention backend: Settings › Compute's, or its own.
const attentionOptions = computed(() => [
  { value: 'inherit', label: t('settings.engineAttentionInherit') },
  ...['auto', 'fast', 'sdpa', 'math', 'sdpa_flash', 'sdpa_efficient', 'sdpa_cudnn', 'flash', 'xformers', 'sage', 'sage3'].map((v) => ({ value: v, label: t(`settings.attentions.${v}`) })),
]);
const attention = computed(() => String(props.settings?.attention ?? 'inherit'));
const configure = (patch: Record<string, unknown>) => actions.configure.mutate({ engine: props.engine.id, patch }, { onError });
// Load options (fp16, the emotion model…), when the engine has any.
const optionsOpen = ref(false);
const loadOptions = computed<Record<string, unknown>>({
  get: () => ({ ...((props.settings?.load_options as Record<string, unknown> | undefined) ?? {}) }),
  set: (value) => configure({ load_options: value }),
});
const hasOptions = computed(() => Object.keys(props.engine.load_options ?? {}).length > 0);
// The dummy engine has nothing to place or tune.
const tunable = computed(() => props.engine.id !== 'dummy');
</script>

<template>
  <AppCard variant="outlined" class="engine">
    <div class="head">
      <div class="titles">
        <span class="type-title-medium">{{ tx(engine.title) }}</span>
        <span class="type-body-small muted">{{ engine.internal ? t('engines.internal') : t('engines.models', { n: engine.models.length }) }} · {{ engine.licence.name }}</span>
      </div>
      <Badge v-if="engine.remote" :value="t('remote.onHost', { host: engine.remote })" tone="neutral" />
      <Badge :value="engine.available ? t('engines.available') : t('engines.unavailable')" :tone="engine.available ? 'primary' : 'warning'" />
    </div>
    <p class="type-body-medium desc">{{ tx(engine.description) }}</p>
    <p v-if="!engine.available" class="type-body-small warn">{{ t('engines.missing', { list: engine.missing.join(', ') }) }}</p>
    <ErrorNotice v-if="error" :error="error" />
    <div v-if="tunable" class="settings">
      <SelectField :model-value="device" :label="t('engines.device')" :options="deviceOptions" @update:model-value="configure({ device: $event })" />
      <SelectField :model-value="dtype" :label="t('engines.dtype')" :options="dtypeOptions" @update:model-value="configure({ dtype: $event })" />
      <SelectField v-if="!engine.internal" :model-value="vramMode" :label="t('settings.engineVramMode')" :options="vramModeOptions" @update:model-value="configure({ vram_mode: $event })" />
      <SelectField :model-value="attention" :label="t('settings.engineAttention')" :options="attentionOptions" @update:model-value="configure({ attention: $event })" />
      <Switch :model-value="engine.enabled" :label="t('engines.enabled')" @update:model-value="configure({ enabled: $event })" />
    </div>
    <ExpansionPanel v-if="tunable && hasOptions" v-model:open="optionsOpen" :label="t('engines.options.title')" :icon="icons.SlidersHorizontal">
      <div class="options">
        <HyperparamsForm v-model="loadOptions" :specs="engine.load_options ?? {}" group="all" />
        <p class="type-body-small muted">{{ t('engines.optionsHint') }}</p>
      </div>
    </ExpansionPanel>
    <ExpansionPanel v-if="tunable && !engine.remote" v-model:open="doctorOpen" :label="t('engines.doctor')" :icon="icons.Stethoscope">
      <ul class="findings">
        <li v-for="(f, i) in findings" :key="i" class="type-body-small" :class="f.level">
          {{ f.message }}<template v-if="f.fix"> — <code>{{ f.fix }}</code></template>
        </li>
      </ul>
    </ExpansionPanel>
    <div class="actions">
      <template v-if="gated">
        <Badge v-if="engine.licence_accepted" :value="t('engines.licenceAccepted')" tone="primary" />
        <AppButton v-else variant="tonal" :icon="icons.ShieldCheck" @click="licenceOpen = true">{{ t('engines.acceptLicence') }}</AppButton>
      </template>
      <span v-if="engine.remote" class="type-body-small muted">{{ t('remote.managedThere', { host: engine.remote, engine: engine.id }) }}</span>
    </div>
    <LicenceDialog v-model:open="licenceOpen" :engine="engine" @accept="(key) => actions.acceptLicence.mutate({ engine: engine.id, licence: key }, { onError })" />
  </AppCard>
</template>

<style scoped>
.engine { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.titles { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.desc { margin: 0; }
.warn { margin: 0; color: var(--md-sys-color-tertiary); }
.settings { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr)); gap: var(--app-space-3); align-items: center; }
.options { display: flex; flex-direction: column; gap: var(--app-space-3); }
.options p { margin: 0; }
.findings { margin: 0; padding-left: var(--app-space-5); display: flex; flex-direction: column; gap: var(--app-space-1); }
.findings .error { color: var(--md-sys-color-error); }
.findings .warning { color: var(--md-sys-color-tertiary); }
.findings code { overflow-wrap: anywhere; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); }
</style>
