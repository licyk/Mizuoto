<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { urls } from '@/api/client';
import { useEngines } from '@/api/queries/engines';
import { useModelSpec, useModels } from '@/api/queries/models';
import { useOutputs } from '@/api/queries/outputs';
import { speak } from '@/api/queries/tts';
import { useVoiceActions } from '@/api/queries/voices';
import type { Voice } from '@/api/types';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import { readyIds, type InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppDialog, AxisPanes, IconButton, SelectField, Tabs, TextArea, TextField, icons } from '@/ui';

/**
 * A new library voice, three ways: from audio (clips, transcripts by speech recognition when left
 * empty), designed (a design model speaks a sample in a described voice; keep the take you
 * like), or a model's preset speaker.
 */
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ created: [Voice] }>();
const { t } = useI18n();
const actions = useVoiceActions();
const models = useModels(() => ({ installed: true }));
const engines = useEngines();
const jobs = useJobsStore();
const player = usePlayerStore();

const TABS = ['audio', 'design', 'preset'] as const;
type Tab = (typeof TABS)[number];
const tab = ref<Tab>('audio');
const tabs = computed(() => TABS.map((v) => ({ value: v, label: t(`voices.tabs.${v}`) })));
const name = ref('');
const language = ref('');
const clips = ref<InputItem[]>([]);
const transcripts = ref<Record<string, string>>({});
const error = ref<unknown>(null);
const busy = ref(false);

watch(open, (o) => {
  if (!o) return;
  name.value = '';
  clips.value = [];
  transcripts.value = {};
  error.value = null;
  takes.value = [];
});

// -- models able to design or with preset speakers ------------------------------------------
const internal = computed(() => new Set((engines.data.value ?? []).filter((e) => e.internal).map((e) => e.id)));
const installed = computed(() => (models.data.value ?? []).filter((m) => !internal.value.has(m.engine)));
const designModel = ref<string | null>(null);
const presetModel = ref<string | null>(null);
const designSpec = useModelSpec(designModel);
const presetSpec = useModelSpec(presetModel);
const withCapability = (...caps: string[]) => installed.value.filter((m) => caps.some((c) => m.capabilities.includes(c))).map((m) => ({ value: m.id, label: m.title }));
const designOptions = computed(() => withCapability('voice_design_freeform', 'voice_design'));
const presetOptions = computed(() => withCapability('preset_speaker'));
watch(designOptions, (opts) => {
  if (!designModel.value && opts.length) designModel.value = opts[0].value;
});
watch(presetOptions, (opts) => {
  if (!presetModel.value && opts.length) presetModel.value = opts[0].value;
});
const canDesign = computed(() => (designSpec.data.value?.voice_modes ?? []).includes('design'));
const speakers = computed(() => presetSpec.data.value?.speakers ?? []);
const speaker = ref<string | null>(null);
watch(speakers, (list) => (speaker.value = list[0] ?? null));
const description = ref('');
const sampleText = ref(t('voices.sampleTextDefault'));

// -- design takes ---------------------------------------------------------------------------
const takes = ref<string[]>([]);
const takeJobs = computed(() => takes.value.map((id) => jobs.byId[id]).filter(Boolean));
const takeOutputs = useOutputs(() => ({}), () => takes.value.length > 0);
const outputOf = (jobId: string) => (takeOutputs.data.value?.items ?? []).find((o) => o.job_id === jobId) ?? null;
async function generate() {
  if (!designModel.value) return;
  error.value = null;
  try {
    const job = await speak({ model: designModel.value, text: sampleText.value, voice_spec: { description: description.value.trim() || null } });
    jobs.upsert(job, 'voices');
    takes.value = [...takes.value, job.id].slice(-4);
  } catch (e) {
    error.value = e;
  }
}

// -- create ---------------------------------------------------------------------------------
const ready = computed(() => {
  if (!name.value.trim()) return false;
  if (tab.value === 'audio') return readyIds(clips.value).length > 0 && !clips.value.some((c) => c.status === 'uploading');
  if (tab.value === 'preset') return !!presetModel.value && !!speaker.value;
  return false;
});
function done(v: Voice) {
  emit('created', v);
  open.value = false;
}
async function create() {
  error.value = null;
  busy.value = true;
  const onError = (e: unknown) => (error.value = e);
  const settle = () => (busy.value = false);
  if (tab.value === 'audio') {
    const items = clips.value.filter((c) => c.status === 'ready' && c.audioId);
    actions.create.mutate(
      {
        name: name.value.trim(),
        audio_ids: items.map((c) => c.audioId as string),
        transcripts: items.map((c) => transcripts.value[c.key]?.trim() || null),
        language: language.value || null,
      },
      { onSuccess: (r) => done(r.voice), onError, onSettled: settle },
    );
  } else if (tab.value === 'preset' && presetModel.value && speaker.value) {
    actions.fromPreset.mutate({ model: presetModel.value, speaker: speaker.value, name: name.value.trim() }, { onSuccess: done, onError, onSettled: settle });
  } else settle();
}
function keep(jobId: string) {
  const out = outputOf(jobId);
  if (!out || !name.value.trim()) return;
  actions.fromOutput.mutate({ output_id: out.id, name: name.value.trim(), description: description.value.trim() }, { onSuccess: done, onError: (e) => (error.value = e) });
}
const languageOptions = computed(() => [{ value: '', label: t('languages.any') }, ...['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'yue'].map((c) => ({ value: c, label: t(`languages.${c}`) }))]);
</script>

<template>
  <AppDialog v-model:open="open" :title="t('voices.new')" width="large" :close-label="t('common.close')">
    <div class="body">
      <Tabs v-model="tab" :tabs="tabs" />
      <TextField v-model="name" :label="t('voices.name')" />
      <AxisPanes :value="tab" :order="TABS">
        <div v-if="tab === 'audio'" key="audio" class="pane">
          <AudioSourceInput v-model="clips" />
          <div v-for="c in clips.filter((x) => x.status === 'ready')" :key="c.key" class="transcript">
            <TextField :model-value="transcripts[c.key] ?? ''" :label="`${t('speak.transcript')}: ${c.name}`" @update:model-value="transcripts = { ...transcripts, [c.key]: String($event ?? '') }" />
          </div>
          <p class="type-body-small muted">{{ t('voices.transcriptsHint') }}</p>
          <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" />
        </div>
        <div v-else-if="tab === 'design'" key="design" class="pane">
          <p v-if="!designOptions.length" class="type-body-small muted">{{ t('voices.noDesignModel') }}</p>
          <SelectField v-else v-model="designModel" :label="t('voices.designModel')" :options="designOptions" />
          <p v-if="designModel && designSpec.isSuccess.value && !canDesign" class="type-body-small muted">{{ t('voices.noDesignModel') }}</p>
          <template v-else>
            <TextArea v-model="description" :label="t('speak.description')" :placeholder="t('speak.descriptionPlaceholder')" :rows="2" :max-length="1000" />
            <TextField v-model="sampleText" :label="t('voices.sampleText')" />
            <div class="row">
              <AppButton variant="tonal" :icon="icons.WandSparkles" :disabled="!description.trim() || !sampleText.trim()" @click="generate">{{ t('voices.generate') }}</AppButton>
            </div>
            <div v-if="takes.length" class="takes">
              <h3 class="type-title-small">{{ t('voices.candidates') }}</h3>
              <template v-for="j in takeJobs" :key="j.id">
                <JobCard v-if="!isFinished(j.state) || j.state !== 'completed'" :job="j" compact />
                <div v-else-if="outputOf(j.id)" class="take">
                  <IconButton
                    :icon="player.isPlaying(j.id) ? icons.Pause : icons.Play"
                    :label="t('results.play')"
                    @click="player.play({ key: j.id, url: urls.output(outputOf(j.id)!.id), label: outputOf(j.id)!.name })"
                  />
                  <span class="type-body-medium grow">{{ outputOf(j.id)!.name }}</span>
                  <AppButton variant="tonal" :disabled="!name.trim()" @click="keep(j.id)">{{ t('voices.keep') }}</AppButton>
                </div>
              </template>
            </div>
          </template>
        </div>
        <div v-else key="preset" class="pane">
          <SelectField v-model="presetModel" :label="t('voices.presetModel')" :options="presetOptions" />
          <SelectField v-if="speakers.length" v-model="speaker" :label="t('speak.speaker')" :options="speakers.map((s) => ({ value: s, label: s }))" />
        </div>
      </AxisPanes>
      <ErrorNotice v-if="error" :error="error" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton v-if="tab !== 'design'" :disabled="!ready" :loading="busy" @click="create">{{ t('voices.create') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.body { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.pane { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.row { display: flex; justify-content: flex-end; }
.takes { display: flex; flex-direction: column; gap: var(--app-space-2); }
.takes h3 { margin: 0; }
.take { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container); min-width: 0; }
.grow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
p { margin: 0; }
</style>
