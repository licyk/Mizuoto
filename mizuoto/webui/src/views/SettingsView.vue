<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useMeta, useSettings, useUpdateSettings, useVersion } from '@/api/queries/app';
import { useComputeInfo } from '@/api/queries/compute';
import { useDoctor } from '@/api/queries/engines';
import { useModels } from '@/api/queries/models';
import type { SettingsView } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import RemoteHosts from '@/components/RemoteHosts.vue';
import { useI18n } from '@/i18n';
import { formatMiB } from '@/format';
import { usePreferencesStore } from '@/stores/preferences';
import { AppButton, Badge, DataList, FilterChip, PathField, SegmentedControl, SelectField, Skeleton, Surface, Switch, Tabs, TextField, TRANSITIONS, icons, useAxisDirection, useSnackbar } from '@/ui';

type Section = 'general' | 'compute' | 'remote' | 'speech' | 'training' | 'downloads' | 'storage' | 'appearance' | 'about';
const { t, tOr, localeOptions } = useI18n();
const prefs = usePreferencesStore();
const settings = useSettings();
const update = useUpdateSettings();
const meta = useMeta();
const version = useVersion();
const compute = useComputeInfo();
const models = useModels(() => ({ installed: true }));
const snackbar = useSnackbar();
const section = ref<Section>('general');
const SECTIONS = ['general', 'compute', 'remote', 'speech', 'training', 'downloads', 'storage', 'appearance', 'about'] as const;
const sections = computed(() => SECTIONS.map((v) => ({ value: v, label: t(`settings.sections.${v}`) })));
// Sections slide the way the tabs moved (shared-axis-x).
const axisDir = useAxisDirection(section, () => SECTIONS);
const error = ref<unknown>(null);
const s = computed<SettingsView | undefined>(() => settings.data.value);

function patch(p: Record<string, unknown>) {
  error.value = null;
  update.mutate(p, { onSuccess: () => snackbar.show(t('settings.saved')), onError: (e) => (error.value = e) });
}
// A value pinned by the application embedding Mizuoto, or by an environment variable, cannot be changed here.
const pinnedBy = (key: string): 'host' | 'env' | null =>
  s.value?.pinned.includes(key) ? 'host' : s.value?.env_overrides.some((e) => e === `MIZUOTO_${key.toUpperCase().replace('.', '__')}`) ? 'env' : null;

// General
const locale = computed({ get: () => prefs.prefs.locale, set: (v) => (prefs.prefs.locale = v as 'auto') });
const token = ref('');
// Compute
const deviceOptions = computed(() => [
  { value: 'auto', label: t('common.auto') },
  ...(compute.data.value?.devices ?? []).map((d) => ({ value: d.id, label: `${d.id} · ${d.name}${d.memory_total_mb ? ` · ${formatMiB(d.memory_total_mb)}` : ''}` })),
]);
// Memory
const vramModeOptions = computed(() => ['auto', 'normal', 'high', 'low', 'none'].map((v) => ({ value: v, label: t(`settings.vramModes.${v}`) })));
// Acceleration: the attention backend, strict mode, the process-wide speed switches.
const ATTENTIONS = ['auto', 'fast', 'sdpa', 'math', 'sdpa_flash', 'sdpa_efficient', 'sdpa_cudnn', 'flash', 'xformers', 'sage', 'sage3'] as const;
const attentionOptions = computed(() => ATTENTIONS.map((v) => ({ value: v, label: t(`settings.attentions.${v}`) })));
const trainAttentionOptions = computed(() => ATTENTIONS.filter((v) => v !== 'sage' && v !== 'sage3').map((v) => ({ value: v, label: t(`settings.attentions.${v}`) })));
const FAST_FEATURES = ['tf32', 'fp16_accumulation', 'cudnn_benchmark', 'sdpa_reduced_precision'] as const;
type FastFeature = (typeof FAST_FEATURES)[number];
const toggleFast = (feature: FastFeature, on: boolean) => {
  const current = (s.value?.compute.fast ?? []) as FastFeature[];
  patch({ compute: { fast: on ? [...current.filter((f) => f !== feature), feature] : current.filter((f) => f !== feature) } });
};
const hostCopiesOptions = computed(() => ['auto', 'on', 'off'].map((v) => ({ value: v, label: t(`settings.hostCopiesModes.${v}`) })));
// torch against the hardware: the doctor's finding, with the command that installs the right build.
const doctor = useDoctor(null, () => section.value === 'compute');
const torchFinding = computed(() => doctor.data.value?.findings.find((f) => f.check === 'torch') ?? null);
const accelFinding = computed(() => doctor.data.value?.findings.find((f) => f.check === 'acceleration') ?? null);
// The finding in this interface's language, from its fields (the server's message is English, for the command line).
interface AccelData {
  attention: string;
  strict: boolean;
  training: string;
  engines: Record<string, string>;
  packages: { backend: string; title: string; version: string | null; runs: Record<string, string | null> }[];
  unusable: string[];
}
const attentionName = (v: string) => tOr(`settings.attentions.${v}`, v);
const accel = computed(() => {
  const d = accelFinding.value?.data as AccelData | null | undefined;
  if (!d) return null;
  const engines = Object.entries(d.engines).map(([e, v]) => `${e} ${attentionName(v)}`);
  return {
    settings: t('settings.accel.settings', { attention: attentionName(d.attention), strict: d.strict ? t('settings.accel.strict') : '', training: attentionName(d.training) }),
    engines: engines.length ? t('settings.accel.engines', { list: engines.join(', ') }) : '',
    packages: d.packages.map((p) => {
      if (!p.version) return { key: p.backend, ok: false, text: t('settings.accel.notInstalled', { name: p.title }) };
      const where = Object.entries(p.runs).map(([device, reason]) => `${device}: ${reason ?? t('settings.accel.runs')}`);
      return { key: p.backend, ok: true, text: t('settings.accel.installed', { name: p.title, version: p.version }) + (where.length ? ` (${where.join('; ')})` : '') };
    }),
    unusable: d.unusable.map((b) => t('settings.accel.unusable', { backend: b })),
  };
});
// Speech
const formatOptions = ['wav', 'flac', 'mp3', 'ogg', 'm4a'].map((v) => ({ value: v, label: v.toUpperCase() }));
const rateOptions = computed(() => [{ value: '', label: t('speak.sampleRateNative') }, ...[16000, 22050, 24000, 44100, 48000].map((r) => ({ value: String(r), label: `${r / 1000} kHz` }))]);
const normalizeOptions = computed(() => ['peak', 'loudness', 'none'].map((v) => ({ value: v, label: t(`speak.normalizes.${v}`) })));
const policyOptions = computed(() => ['trim', 'reject'].map((v) => ({ value: v, label: t(`settings.referencePolicies.${v}`) })));
const modelOptions = computed(() => [{ value: '', label: t('common.none') }, ...(models.data.value ?? []).map((m) => ({ value: m.id, label: m.title }))]);
const asrOptions = computed(() => (models.data.value ?? []).filter((m) => m.engine === 'asr').map((m) => ({ value: m.id, label: m.title })));
// Training
const trainDeviceOptions = computed(() => [
  { value: '', label: t('settings.trainDeviceEngine') },
  { value: 'cpu', label: 'CPU' },
  ...(compute.data.value?.devices ?? []).filter((d) => d.id !== 'cpu').map((d) => ({ value: d.id, label: `${d.id} · ${d.name}` })),
]);
const trainMemoryOptions = computed(() => (['standard', 'low', 'lowest'] as const).map((v) => ({ value: v, label: t(`settings.trainMemories.${v}`) })));
const precisionOptions = computed(() => (['auto', 'fp32', 'fp16', 'bf16'] as const).map((v) => ({ value: v, label: v === 'auto' ? t('common.auto') : v })));
// Downloads
const sourceOptions = computed(() => (['huggingface', 'hf-mirror', 'modelscope', 'custom'] as const).map((v) => ({ value: v, label: t(`settings.sources.${v}`) })));
const repositoryOptions = computed(() => (['mizuoto-model', 'official'] as const).map((v) => ({ value: v, label: t(`settings.repositories.${v}`) })));
const endpoint = ref('');
watch(s, (v) => v && (endpoint.value = v.downloads.endpoint), { immediate: true });
// Storage
const FOLDERS = ['models_dir', 'voices_dir', 'outputs_dir', 'cache_dir'] as const;
const folders = ref<Record<string, string>>({});
const browseRoots = ref('');
watch(
  s,
  (v) => {
    if (!v) return;
    folders.value = Object.fromEntries(FOLDERS.map((k) => [k, (v.paths as Record<string, unknown>)[k] as string]));
    browseRoots.value = v.paths.browse_roots.join('\n');
  },
  { immediate: true },
);
function saveFolders() {
  patch({ paths: { ...folders.value, browse_roots: browseRoots.value.split('\n').map((x) => x.trim()).filter(Boolean) } });
}
// Appearance
const themes = computed(() => (['light', 'dark', 'system'] as const).map((v) => ({ value: v, label: t(`settings.themes.${v}`) })));
const motions = computed(() => (['system', 'reduced'] as const).map((v) => ({ value: v, label: t(`settings.motions.${v}`) })));
const contrastOptions = computed(() => [
  { value: '0', label: t('settings.contrastLevels.standard') },
  { value: '0.5', label: t('settings.contrastLevels.medium') },
  { value: '1', label: t('settings.contrastLevels.high') },
]);
const contrast = computed({ get: () => String(prefs.prefs.contrast), set: (v) => (prefs.prefs.contrast = Number(v)) });
const theme = computed({ get: () => prefs.prefs.theme, set: (v) => (prefs.prefs.theme = v) });
const motion = computed({ get: () => prefs.prefs.motion, set: (v) => (prefs.prefs.motion = v) });
const previewShown = ref(true);
</script>

<template>
  <div class="settings">
    <Tabs v-model="section" :tabs="sections" />
    <ErrorNotice v-if="error" :error="error" />
    <ErrorNotice v-if="settings.error.value" :error="settings.error.value" />
    <div v-if="s" class="panes" :style="{ '--axis-dir': axisDir }">
      <Transition :name="TRANSITIONS.sharedAxisX">
        <Surface :key="section" :level="0" class="panel">
          <template v-if="section === 'general'">
            <SelectField v-model="locale" :label="t('settings.language')" :options="localeOptions" />
            <Switch :model-value="s.server.open_browser" :label="t('settings.openBrowser')" @update:model-value="patch({ server: { open_browser: $event } })" />
            <div class="row">
              <TextField :model-value="s.server.host" :label="t('settings.address')" :supporting-text="t('settings.restartNote')" @change="patch({ server: { host: $event } })" />
              <TextField type="number" :model-value="s.server.port" :label="t('settings.port')" @change="patch({ server: { port: Number($event) } })" />
            </div>
            <div class="row center">
              <Badge :value="s.server.access_token_configured ? t('settings.tokenSet') : t('settings.tokenNone')" :tone="s.server.access_token_configured ? 'primary' : 'neutral'" />
              <TextField v-model="token" class="grow" type="password" :label="t('settings.accessToken')" autocomplete="new-password" />
              <AppButton variant="tonal" :disabled="!token" @click="patch({ server: { access_token: token } }), (token = '')">{{ t('settings.setToken') }}</AppButton>
              <AppButton v-if="s.server.access_token_configured" variant="text" @click="patch({ server: { access_token: null } })">{{ t('settings.clearToken') }}</AppButton>
            </div>
          </template>
          <template v-else-if="section === 'compute'">
            <SelectField :model-value="s.compute.default_device" :label="t('settings.defaultDevice')" :options="deviceOptions" @update:model-value="patch({ compute: { default_device: $event } })" />
            <div class="row">
              <TextField type="number" :min="0" :model-value="s.compute.unload_after_minutes" :label="t('settings.unloadAfter')" @change="patch({ compute: { unload_after_minutes: Number($event) } })" />
              <TextField type="number" :min="1" :model-value="s.compute.max_resident_models" :label="t('settings.maxModels')" @change="patch({ compute: { max_resident_models: Number($event) } })" />
              <TextField type="number" :min="1" :model-value="s.compute.gpu_jobs" :label="t('settings.gpuJobs')" @change="patch({ compute: { gpu_jobs: Number($event) } })" />
              <TextField
                type="number"
                :min="1"
                :max="16"
                :model-value="s.compute.inference_jobs"
                :label="t('settings.inferenceJobs')"
                :supporting-text="t('settings.inferenceJobsHint')"
                @change="patch({ compute: { inference_jobs: Number($event) } })"
              />
            </div>
            <Switch :model-value="s.compute.compile" :label="t('settings.compile')" :supporting-text="t('settings.compileHint')" @update:model-value="patch({ compute: { compile: $event } })" />
            <Switch :model-value="s.compute.cuda_graphs" :label="t('settings.cudaGraphs')" :supporting-text="t('settings.cudaGraphsHint')" @update:model-value="patch({ compute: { cuda_graphs: $event } })" />
            <h3 class="type-title-small">{{ t('settings.acceleration') }}</h3>
            <SelectField :model-value="s.compute.attention" :label="t('settings.attention')" :supporting-text="t('settings.attentionHint')" :options="attentionOptions" @update:model-value="patch({ compute: { attention: $event } })" />
            <Switch :model-value="s.compute.attention_strict" :label="t('settings.attentionStrict')" :supporting-text="t('settings.attentionStrictHint')" @update:model-value="patch({ compute: { attention_strict: $event } })" />
            <div class="row chips">
              <FilterChip v-for="f in FAST_FEATURES" :key="f" :label="t(`settings.fastFeatures.${f}`)" :model-value="s.compute.fast.includes(f)" @update:model-value="toggleFast(f, $event)" />
            </div>
            <p class="type-body-small muted">{{ t('settings.fastHint') }}</p>
            <div v-if="accel" class="accel type-body-small">
              <span class="muted">{{ accel.settings }}</span>
              <span v-if="accel.engines" class="muted">{{ accel.engines }}</span>
              <ul class="packages">
                <li v-for="p in accel.packages" :key="p.key" :class="p.ok ? '' : 'muted'">{{ p.text }}</li>
              </ul>
              <span v-for="u in accel.unusable" :key="u" class="warn">{{ u }}</span>
            </div>
            <p v-else-if="accelFinding" class="type-body-small" :class="accelFinding.level === 'ok' ? 'muted' : 'warn'">{{ accelFinding.message }}</p>
            <code v-if="accelFinding?.fix" class="command">{{ accelFinding.fix }}</code>
            <h3 class="type-title-small">{{ t('settings.memory') }}</h3>
            <SelectField :model-value="s.compute.vram_mode" :label="t('settings.vramMode')" :options="vramModeOptions" @update:model-value="patch({ compute: { vram_mode: $event } })" />
            <div class="row">
              <TextField type="number" :min="0" :model-value="s.compute.vram_reserve_mb" :label="t('settings.vramReserve')" @change="patch({ compute: { vram_reserve_mb: Number($event) } })" />
              <TextField type="number" :min="0" :model-value="s.compute.vram_limit_mb" :label="t('settings.vramLimit')" @change="patch({ compute: { vram_limit_mb: Number($event) } })" />
              <TextField type="number" :min="0" :model-value="s.compute.ram_reserve_mb ?? ''" :label="t('settings.ramReserve')" @change="patch({ compute: { ram_reserve_mb: $event === '' ? null : Number($event) } })" />
            </div>
            <div class="row">
              <SelectField :model-value="s.compute.host_copies" :label="t('settings.hostCopies')" :options="hostCopiesOptions" @update:model-value="patch({ compute: { host_copies: $event } })" />
              <TextField type="number" :min="0" :model-value="s.compute.max_pinned_mb" :label="t('settings.maxPinned')" @change="patch({ compute: { max_pinned_mb: Number($event) } })" />
              <TextField type="number" :min="-1" :max="8" :model-value="s.compute.async_offload" :label="t('settings.asyncOffload')" @change="patch({ compute: { async_offload: Number($event) } })" />
            </div>
            <Switch :model-value="s.compute.oom_retry" :label="t('settings.oomRetry')" :supporting-text="t('settings.oomRetryHint')" @update:model-value="patch({ compute: { oom_retry: $event } })" />
            <Switch :model-value="s.compute.oom_fallback" :label="t('settings.oomFallback')" :supporting-text="t('settings.oomFallbackHint')" @update:model-value="patch({ compute: { oom_fallback: $event } })" />
            <DataList :rows="(compute.data.value?.devices ?? []).map((d) => ({ label: d.id, value: `${d.name}${d.memory_total_mb ? ' · ' + formatMiB(d.memory_used_mb ?? 0) + ' / ' + formatMiB(d.memory_total_mb) : ''}` }))" />
            <template v-if="doctor.data.value">
              <DataList :rows="[{ label: 'torch', value: doctor.data.value.torch ? `${doctor.data.value.torch} (${doctor.data.value.torch_build ?? '?'})` : t('settings.torchMissing') }]" />
              <p v-if="torchFinding && torchFinding.level !== 'ok'" class="type-body-small warn">{{ torchFinding.message }}</p>
              <code v-if="torchFinding?.fix" class="command">{{ torchFinding.fix }}</code>
            </template>
          </template>
          <template v-else-if="section === 'remote'">
            <RemoteHosts />
          </template>
          <template v-else-if="section === 'speech'">
            <SelectField :model-value="s.tts.default_model" :label="t('settings.defaultModel')" :options="modelOptions" @update:model-value="patch({ tts: { default_model: $event ?? '' } })" />
            <div class="row">
              <SelectField :model-value="s.tts.output_format" :label="t('settings.outputFormat')" :options="formatOptions" @update:model-value="patch({ tts: { output_format: $event } })" />
              <SelectField :model-value="s.tts.output_sample_rate ? String(s.tts.output_sample_rate) : ''" :label="t('settings.outputSampleRate')" :options="rateOptions" @update:model-value="patch({ tts: { output_sample_rate: $event ? Number($event) : null } })" />
              <SelectField :model-value="s.tts.normalize" :label="t('settings.normalize')" :options="normalizeOptions" @update:model-value="patch({ tts: { normalize: $event } })" />
            </div>
            <Switch :model-value="s.tts.auto_transcribe" :label="t('settings.autoTranscribe')" @update:model-value="patch({ tts: { auto_transcribe: $event } })" />
            <SelectField v-if="asrOptions.length" :model-value="s.tts.asr_model" :label="t('settings.asrModel')" :options="asrOptions" @update:model-value="patch({ tts: { asr_model: $event } })" />
            <SelectField :model-value="s.tts.reference_policy" :label="t('settings.referencePolicy')" :options="policyOptions" @update:model-value="patch({ tts: { reference_policy: $event } })" />
            <Switch :model-value="s.tts.strict_params" :label="t('settings.strictParams')" :supporting-text="t('settings.strictHint')" @update:model-value="patch({ tts: { strict_params: $event } })" />
            <Switch :model-value="s.tts.micro_batch" :label="t('settings.microBatch')" :supporting-text="t('settings.microBatchHint')" @update:model-value="patch({ tts: { micro_batch: $event } })" />
          </template>
          <template v-else-if="section === 'training'">
            <SelectField :model-value="s.training.devices[0] ?? ''" :label="t('settings.trainDevice')" :supporting-text="t('settings.trainDeviceHint')" :options="trainDeviceOptions" @update:model-value="patch({ training: { devices: $event ? [$event] : [] } })" />
            <div class="row">
              <SelectField :model-value="s.training.precision" :label="t('settings.trainPrecision')" :options="precisionOptions" @update:model-value="patch({ training: { precision: $event } })" />
              <SelectField :model-value="s.training.memory" :label="t('settings.trainMemory')" :options="trainMemoryOptions" @update:model-value="patch({ training: { memory: $event } })" />
              <SelectField :model-value="s.training.attention" :label="t('settings.trainAttention')" :options="trainAttentionOptions" @update:model-value="patch({ training: { attention: $event } })" />
              <TextField type="number" :min="1" :model-value="s.training.keep_checkpoints" :label="t('settings.keepCheckpoints')" @change="patch({ training: { keep_checkpoints: Number($event) } })" />
            </div>
            <Switch :model-value="s.training.previews" :label="t('settings.trainPreviews')" :supporting-text="t('settings.trainPreviewsHint')" @update:model-value="patch({ training: { previews: $event } })" />
            <Switch :model-value="s.training.deterministic" :label="t('settings.trainDeterministic')" :supporting-text="t('settings.trainDeterministicHint')" @update:model-value="patch({ training: { deterministic: $event } })" />
          </template>
          <template v-else-if="section === 'downloads'">
            <SelectField :model-value="s.downloads.source" :label="t('settings.source')" :options="sourceOptions" @update:model-value="patch({ downloads: { source: $event } })" />
            <TextField v-if="s.downloads.source === 'custom'" v-model="endpoint" :label="t('settings.endpoint')" @change="patch({ downloads: { endpoint } })" />
            <SelectField :model-value="s.downloads.repository" :label="t('settings.repository')" :supporting-text="t('settings.repositoryHint')" :options="repositoryOptions" @update:model-value="patch({ downloads: { repository: $event } })" />
            <Switch :model-value="s.downloads.verify_checksums" :label="t('settings.verifyChecksums')" @update:model-value="patch({ downloads: { verify_checksums: $event } })" />
            <TextField type="number" :min="1" :max="8" :model-value="s.downloads.concurrency" :label="t('settings.concurrency')" @change="patch({ downloads: { concurrency: Number($event) } })" />
          </template>
          <template v-else-if="section === 'storage'">
            <h3 class="type-title-small sub">{{ t('settings.folders') }}</h3>
            <PathField
              v-for="key in FOLDERS"
              :key="key"
              v-model="folders[key]"
              :label="t(`settings.${key.replace('_dir', 'Dir')}`)"
              :supporting-text="pinnedBy(`paths.${key}`) ? t(pinnedBy(`paths.${key}`) === 'host' ? 'settings.pinnedByHost' : 'settings.pinned') : t('settings.defaultFolder', { path: (s.resolved_paths as Record<string, unknown>)[key] as string })"
            />
            <label class="type-body-medium">{{ t('settings.browseRoots') }}</label>
            <textarea v-model="browseRoots" class="area type-body-medium" rows="4" :placeholder="s.resolved_paths.browse_roots.join('\n')" />
            <span class="type-body-small muted">{{ t('settings.browseRootsHint') }}</span>
            <TextField type="number" :min="0" :model-value="s.outputs.keep_days" :label="t('settings.keepOutputs')" @change="patch({ outputs: { keep_days: Number($event) } })" />
            <div class="actions"><AppButton :icon="icons.Save" @click="saveFolders">{{ t('common.save') }}</AppButton></div>
          </template>
          <template v-else-if="section === 'appearance'">
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.theme') }}</span>
              <SegmentedControl v-model="theme" :options="themes" />
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.color') }}</span>
              <!-- Any colour: every role is generated from it. -->
              <input v-model="prefs.prefs.sourceColor" type="color" class="color" :aria-label="t('settings.color')" />
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.contrast') }}</span>
              <SelectField v-model="contrast" :options="contrastOptions" />
            </div>
            <div class="field-row">
              <span class="type-body-large">{{ t('settings.motion') }}</span>
              <SegmentedControl v-model="motion" :options="motions" />
            </div>
            <AppButton variant="text" class="start" @click="previewShown = !previewShown">{{ t('settings.motionPreview') }}</AppButton>
            <Transition name="container"><div v-if="previewShown" class="preview type-body-medium">{{ t('settings.motionPreview') }}</div></Transition>
          </template>
          <template v-else>
            <DataList
              :rows="[
                { label: 'Mizuoto', value: version.data.value?.version ?? '—' },
                { label: t('settings.dataDir'), value: meta.data.value?.data_dir || '—', mono: true },
                { label: t('settings.settingsFile'), value: s.settings_file, mono: true },
                { label: t('settings.defaultDevice'), value: compute.data.value?.backend ?? '—' },
              ]"
            />
            <p class="type-body-small muted">{{ t('settings.licences') }}</p>
          </template>
        </Surface>
      </Transition>
    </div>
    <!-- The settings are one request; until it answers, the panel's shape rather than an empty page. -->
    <Surface v-else-if="!settings.error.value" :level="0" class="panel" role="status" aria-busy="true" :aria-label="t('common.loading')">
      <Skeleton v-for="row in 4" :key="row" height="56px" shape="medium" />
    </Surface>
  </div>
</template>

<style scoped>
.settings { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4); min-width: 0; }
/* The leaving section is taken out of the flow (shared-axis-x) and slides inside this box. */
.panes { position: relative; overflow-x: clip; }
.panel { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-5); min-width: 0; }
.row { display: flex; flex-wrap: wrap; gap: var(--app-space-3); align-items: flex-start; }
.row:not(.center) > * { flex: 1 1 280px; min-width: 0; }
.row.center { align-items: center; }
.row.chips > * { flex: 0 0 auto; }
.row.center > .grow { flex: 1 1 280px; min-width: 0; }
.sub { margin: 0; }
.warn { margin: 0; color: var(--md-sys-color-tertiary); }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
.accel { display: flex; flex-direction: column; gap: var(--app-space-1); }
.packages { margin: 0; padding-inline-start: var(--app-space-5); }
.command { display: block; overflow-wrap: anywhere; user-select: all; }
.actions { display: flex; justify-content: flex-end; }
.area { width: 100%; padding: var(--app-space-2) var(--app-space-3); border: 1px solid var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-extra-small); background: transparent; color: var(--md-sys-color-on-surface); font: inherit; resize: vertical; }
.field-row { display: flex; align-items: center; justify-content: space-between; gap: var(--app-space-4); min-height: 56px; flex-wrap: wrap; }
.start { align-self: flex-start; }
.color { width: 56px; height: 40px; padding: 0; border: 1px solid var(--md-sys-color-outline); border-radius: var(--md-sys-shape-corner-small); background: transparent; cursor: pointer; }
.preview { padding: var(--app-space-4); border-radius: var(--md-sys-shape-corner-large); background: var(--md-sys-color-primary-container); color: var(--md-sys-color-on-primary-container); }
</style>
