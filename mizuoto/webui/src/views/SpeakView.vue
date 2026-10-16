<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, ref, watch } from 'vue';
import { api, unwrap } from '@/api/client';
import { useSettings } from '@/api/queries/app';
import { transcribeAudio } from '@/api/queries/audio';
import { useModelSpec, useModels } from '@/api/queries/models';
import { useEngines } from '@/api/queries/engines';
import { fetchRerun } from '@/api/queries/outputs';
import { usePresets } from '@/api/queries/presets';
import { planSpeech, speak } from '@/api/queries/tts';
import { useModelVoiceFits, useVoiceActions } from '@/api/queries/voices';
import type { EmotionInput, Output, Preset, SpeakRequest, SpeakRequestOut, SynthesisPlan, VoiceMode } from '@/api/types';
import { StreamPlayer, type StreamStatus } from '@/audio/streamPlayer';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import EmotionPanel from '@/components/EmotionPanel.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobsTray from '@/components/JobsTray.vue';
import ModelGate from '@/components/ModelGate.vue';
import ModelPicker from '@/components/ModelPicker.vue';
import ParamsPanel from '@/components/ParamsPanel.vue';
import PlanDialog from '@/components/PlanDialog.vue';
import PresetMenu from '@/components/PresetMenu.vue';
import ResultsList from '@/components/ResultsList.vue';
import ScriptEditor from '@/components/ScriptEditor.vue';
import StreamPlayerBar from '@/components/StreamPlayerBar.vue';
import VoicePicker from '@/components/VoicePicker.vue';
import { inputKey, type InputItem } from '@/components/inputs';
import { useSessionOutputs } from '@/components/sessionOutputs';
import { useI18n } from '@/i18n';
import { useHandoffStore } from '@/stores/handoff';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { usePreferencesStore } from '@/stores/preferences';
import { AppButton, AppDialog, AxisPanes, Badge, Chip, FieldRow, IconButton, PageFooter, SegmentedControl, SelectField, Surface, Switch, TextArea, TextField, Tooltip, icons, useSnackbar } from '@/ui';

const VOICE_KINDS = ['library', 'clip', 'preset', 'design', 'auto'] as const;
type VoiceKind = (typeof VOICE_KINDS)[number];

const { t, tOr } = useI18n();
const prefs = usePreferencesStore();
const jobs = useJobsStore();
const handoff = useHandoffStore();
const settings = useSettings();
const voiceActions = useVoiceActions();
const snackbar = useSnackbar();

// -- model --------------------------------------------------------------------------------
const model = ref<string | null>(prefs.prefs.speakModel);
watch(model, (m) => (prefs.prefs.speakModel = m));
// Nothing chosen yet: the default model, else the first ready one (a real engine before the demo).
const allModels = useModels();
const engines = useEngines();
watch(
  [() => allModels.data.value, () => engines.data.value, () => settings.data.value],
  ([list, engineList, s]) => {
    if (model.value || !list || !engineList || !s) return;
    const ready = new Set(engineList.filter((e) => !e.internal && e.enabled && e.licence_accepted && e.available).map((e) => e.id));
    const usable = list.filter((m) => ready.has(m.engine) && ['installed', 'builtin'].includes(m.state));
    model.value = usable.find((m) => m.id === s.tts.default_model)?.id ?? usable.find((m) => m.state === 'installed')?.id ?? usable[0]?.id ?? null;
  },
  { immediate: true },
);
const spec = useModelSpec(model);
const caps = computed(() => new Set(spec.data.value?.capabilities ?? []));
const modes = computed(() => new Set<VoiceMode>(spec.data.value?.voice_modes ?? []));

// -- voice --------------------------------------------------------------------------------
const voiceKinds = computed<VoiceKind[]>(() => {
  const m = modes.value;
  const out: VoiceKind[] = [];
  if (m.has('clone') || m.has('clone_no_transcript') || m.has('preset') || m.has('design')) out.push('library');
  if (m.has('clone') || m.has('clone_no_transcript')) out.push('clip');
  if (m.has('preset')) out.push('preset');
  if (m.has('design')) out.push('design');
  if (m.has('auto')) out.push('auto');
  return out;
});
const voiceKind = ref<VoiceKind>('library');
const fits = useModelVoiceFits(model);
// For a newly chosen model, once its spec and the library's fits are in: the library voice when it
// suits the model, else the model's natural way to get a voice (its preset speakers, a description,
// a clip), else whatever it offers. Chosen once per model; the user's choice then stands.
let chosenFor: string | null = null;
watch([voiceKinds, () => fits.data.value, () => fits.isFetching.value], ([kinds, fitList, fetching]) => {
  if (!kinds.length || !model.value) return;
  if (chosenFor === model.value) {
    if (!kinds.includes(voiceKind.value)) voiceKind.value = kinds[0];
    return;
  }
  if (voiceId.value && kinds.includes('library') && (fetching || !fitList)) return;
  chosenFor = model.value;
  const libraryFits = !!voiceId.value && (fitList ?? []).some((f) => f.voice_id === voiceId.value && f.usable);
  if (kinds.includes('library') && libraryFits) voiceKind.value = 'library';
  else voiceKind.value = (['preset', 'design', 'clip', 'library', 'auto'] as VoiceKind[]).find((k) => kinds.includes(k)) ?? kinds[0];
});
const kindOptions = computed(() => voiceKinds.value.map((k) => ({ value: k, label: t(`speak.voiceKinds.${k}`) })));

const voiceId = ref<string | null>(prefs.prefs.speakVoice);
watch(voiceId, (v) => (prefs.prefs.speakVoice = v));
const clip = ref<InputItem[]>([]);
const transcript = ref('');
const transcribing = ref(false);
const speaker = ref<string | null>(null);
const speakers = computed(() => spec.data.value?.speakers ?? []);
watch(speakers, (list) => {
  if (list.length && (!speaker.value || !list.includes(speaker.value))) speaker.value = list[0];
});
const speakerOptions = computed(() => speakers.value.map((s) => ({ value: s, label: s })));
const description = ref('');
const attributes = ref<Record<string, string>>({});
const designAttributes = computed(() => spec.data.value?.design_attributes ?? null);
const freeform = computed(() => caps.value.has('voice_design_freeform'));

// Transcribing needs the speech recognition model: offered for download in the button's place when missing.
const asrModel = computed(() => settings.data.value?.tts.asr_model ?? null);
function onClipAdded(item: InputItem) {
  if (item.transcript) transcript.value = item.transcript;
}
async function transcribe() {
  const audioId = clip.value[0]?.audioId;
  if (!audioId) return;
  transcribing.value = true;
  try {
    transcript.value = (await transcribeAudio(audioId, language.value === 'auto' ? null : language.value)).text;
  } catch (e) {
    error.value = e;
  } finally {
    transcribing.value = false;
  }
}

// -- text, style, parameters, output ------------------------------------------------------
const text = ref('');
const language = ref(prefs.prefs.speakLanguage || 'auto');
watch(language, (l) => (prefs.prefs.speakLanguage = l));
const batch = ref(false);
const instruct = ref('');
const emotion = ref<EmotionInput | null>(null);
const emotionPanel = ref<InstanceType<typeof EmotionPanel> | null>(null);
const emotionWays = computed(() => ({ vector: caps.value.has('emotion_vector'), audio: caps.value.has('emotion_audio'), text: caps.value.has('emotion_text') }));
const hasStyle = computed(() => caps.value.has('instruct') || Object.values(emotionWays.value).some(Boolean));
const params = ref<Record<string, unknown>>({});
const options = ref<Record<string, unknown>>({});
watch(model, () => {
  params.value = {};
  options.value = {};
  emotionPanel.value?.reset();
});

// -- presets: a model's default preset is applied when the model (and voice) is picked ----------
const presets = usePresets(model);
function applyPreset(p: Preset) {
  const { seed: s, ...rest } = p.params as Record<string, unknown>;
  params.value = { ...rest };
  options.value = { ...p.engine_options };
  if (typeof s === 'number') seed.value = s;
}
let defaultsFor: string | null = null;
watch(
  () => presets.data.value,
  (list) => {
    if (!list || !model.value || defaultsFor === model.value) return;
    defaultsFor = model.value;
    const own = list.filter((p) => p.model === model.value && p.is_default);
    const chosen = own.find((p) => voiceKind.value === 'library' && p.voice_id === voiceId.value) ?? own.find((p) => !p.voice_id);
    if (chosen) applyPreset(chosen);
  },
);
const seed = ref<number | null>(null);
const takesSeed = computed(() => (spec.data.value?.params ?? []).some((p) => p.name === 'seed'));
const rollSeed = () => (seed.value = Math.floor(Math.random() * 2 ** 31));
const format = ref<string | null>(null);
const sampleRate = ref<string>('');
const normalize = ref<string | null>(null);
const formatOptions = ['wav', 'flac', 'mp3', 'ogg', 'm4a'].map((v) => ({ value: v, label: v.toUpperCase() }));
const rateOptions = computed(() => [{ value: '', label: t('speak.sampleRateNative') }, ...[16000, 22050, 24000, 44100, 48000].map((r) => ({ value: String(r), label: `${r / 1000} kHz` }))]);
const normalizeOptions = computed(() => ['peak', 'loudness', 'none'].map((v) => ({ value: v, label: t(`speak.normalizes.${v}`) })));
const chosenFormat = computed(() => format.value ?? settings.data.value?.tts.output_format ?? 'wav');
const chosenNormalize = computed(() => normalize.value ?? settings.data.value?.tts.normalize ?? 'peak');

// -- the request ----------------------------------------------------------------------------
const voiceProblem = computed(() => {
  if (voiceKind.value === 'library' && !voiceId.value) return t('speak.noVoice');
  if (voiceKind.value === 'clip' && !clip.value.some((c) => c.status === 'ready')) return t('speak.noVoice');
  if (voiceKind.value === 'preset' && !speaker.value) return t('speak.noVoice');
  if (voiceKind.value === 'design' && !description.value.trim() && !Object.keys(attributes.value).length) return t('speak.noVoice');
  return '';
});
const style = computed(() => {
  const i = caps.value.has('instruct') && instruct.value.trim() ? instruct.value.trim() : null;
  return i || emotion.value ? { instruct: i, emotion: emotion.value } : null;
});
const request = computed<SpeakRequest | null>(() => {
  if (!model.value || !spec.data.value) return null;
  const lines = text.value.split('\n').map((l) => l.trim()).filter(Boolean);
  const req: SpeakRequest = {
    model: model.value,
    text: batch.value ? '' : text.value,
    texts: batch.value ? lines : null,
    language: language.value === 'auto' ? null : language.value,
    params: { ...params.value, ...(seed.value !== null && takesSeed.value ? { seed: seed.value } : {}) },
    engine_options: { ...options.value },
    output: { format: chosenFormat.value as 'wav', sample_rate: sampleRate.value ? Number(sampleRate.value) : null, normalize: normalize.value as 'peak' | null },
    style: style.value,
  };
  if (voiceKind.value === 'library') req.voice = voiceId.value;
  else if (voiceKind.value === 'clip') {
    const c = clip.value.find((i) => i.status === 'ready' && i.audioId);
    if (c) req.voice_spec = { references: [{ audio_id: c.audioId, transcript: transcript.value.trim() || null }] };
  } else if (voiceKind.value === 'preset') req.voice_spec = { preset: speaker.value };
  else if (voiceKind.value === 'design') req.voice_spec = { description: description.value.trim() || null, attributes: Object.keys(attributes.value).length ? attributes.value : null };
  return req;
});
const textProblem = computed(() => (text.value.trim() ? '' : t('speak.noText')));
const canSpeak = computed(() => !!request.value && !voiceProblem.value && !textProblem.value && !clip.value.some((c) => c.status === 'uploading'));

const error = ref<unknown>(null);
const busy = ref(false);
const sessionJobs = computed(() => jobs.fromScreen('speak').filter((j) => !isFinished(j.state)));
const outputs = useSessionOutputs('speak');

// -- streaming: hear it as it is made ---------------------------------------------------------
const stream = computed({ get: () => prefs.prefs.speakStream, set: (v: boolean) => (prefs.prefs.speakStream = v) });
const granularity = computed(() => spec.data.value?.stream_granularity ?? 'emulated');
const streamStatus = ref<StreamStatus | null>(null);
let player: StreamPlayer | null = null;
async function runStreaming(req: SpeakRequest) {
  player?.close();
  const p = new StreamPlayer();
  player = p;
  p.onStatus = (s) => {
    if (player !== p) return;
    streamStatus.value = s;
    if (s.state === 'failed' && s.error) error.value = s.error;
    // The result is in Results by then; the bar goes a moment later.
    if (s.state === 'done' || s.state === 'stopped') setTimeout(() => player === p && (streamStatus.value = null), 4000);
  };
  const jobId = await p.start(req);
  jobs.upsert(await unwrap(api.GET('/api/v1/jobs/{job_id}', { params: { path: { job_id: jobId } } })), 'speak');
}
function stopStreaming() {
  player?.cancel();
}
onBeforeUnmount(() => player?.close());

async function run() {
  if (!canSpeak.value || !request.value || busy.value) return;
  error.value = null;
  busy.value = true;
  try {
    if (stream.value) await runStreaming(request.value);
    else jobs.upsert(await speak(request.value), 'speak');
  } catch (e) {
    error.value = e;
  } finally {
    busy.value = false;
  }
}
const plan = ref<SynthesisPlan | null>(null);
const planOpen = ref(false);
async function showPlan() {
  if (!request.value) return;
  error.value = null;
  try {
    plan.value = await planSpeech(request.value);
    planOpen.value = true;
  } catch (e) {
    error.value = e;
  }
}

// -- results ------------------------------------------------------------------------------
async function useAsReference(o: Output) {
  try {
    const f = await unwrap(api.POST('/api/v1/audio/files/from-output/{output_id}', { params: { path: { output_id: o.id } } }));
    clip.value = [{ key: inputKey(), name: o.name, audioId: f.id, duration: f.duration, status: 'ready' }];
    transcript.value = o.text ?? '';
    if (voiceKinds.value.includes('clip')) voiceKind.value = 'clip';
    snackbar.show(t('speak.referenceAdded'));
  } catch (e) {
    error.value = e;
  }
}
const keepOutput = ref<Output | null>(null);
const keepName = ref('');
function saveVoice(o: Output) {
  keepOutput.value = o;
  keepName.value = '';
}
function confirmKeep() {
  const o = keepOutput.value;
  if (!o || !keepName.value.trim()) return;
  voiceActions.fromOutput.mutate(
    { output_id: o.id, name: keepName.value.trim() },
    { onSuccess: (v) => snackbar.show(t('voices.created', { name: v.name })), onError: (e) => (error.value = e) },
  );
  keepOutput.value = null;
}
const keepOpen = computed({ get: () => !!keepOutput.value, set: (v) => (keepOutput.value = v ? keepOutput.value : null) });

function apply(r: SpeakRequestOut) {
  model.value = r.model;
  chosenFor = r.model;
  if (r.texts?.length) {
    batch.value = true;
    text.value = r.texts.join('\n');
  } else {
    batch.value = false;
    text.value = r.text;
  }
  language.value = r.language ?? 'auto';
  const { seed: s, ...rest } = (r.params ?? {}) as Record<string, unknown>;
  params.value = Object.fromEntries(Object.entries(rest).filter(([, v]) => v !== null && v !== undefined));
  seed.value = typeof s === 'number' ? s : null;
  // Applied after the model's spec arrives (its watcher clears the options).
  setTimeout(() => (options.value = { ...(r.engine_options ?? {}) }), 0);
  instruct.value = r.style?.instruct ?? '';
  defaultsFor = r.model;
  if (r.voice) {
    voiceKind.value = 'library';
    voiceId.value = r.voice;
  } else if (r.voice_spec?.preset) {
    voiceKind.value = 'preset';
    speaker.value = r.voice_spec.preset;
  } else if (r.voice_spec?.description || r.voice_spec?.attributes) {
    voiceKind.value = 'design';
    description.value = r.voice_spec.description ?? '';
    attributes.value = { ...(r.voice_spec.attributes ?? {}) };
  } else if (r.voice_spec?.references?.length) {
    const ref0 = r.voice_spec.references[0];
    if (ref0.audio_id) {
      voiceKind.value = 'clip';
      clip.value = [{ key: inputKey(), name: ref0.path ?? ref0.audio_id, audioId: ref0.audio_id, status: 'ready' }];
      transcript.value = ref0.transcript ?? '';
    }
  } else voiceKind.value = 'auto';
}
async function rerun(o: Output) {
  try {
    apply(await fetchRerun(o.id));
    snackbar.show(t('speak.rerunLoaded'));
  } catch (e) {
    error.value = e;
  }
}

// "Run again" from History, "Use in Speak" from Voices, "Use as reference" from elsewhere.
onActivated(() => {
  const r = handoff.takeRequest();
  if (r) apply(r);
  const v = handoff.takeVoice();
  if (v) {
    chosenFor = model.value;
    voiceKind.value = 'library';
    voiceId.value = v;
  }
  const c = handoff.takeReference();
  if (c) {
    chosenFor = model.value;
    clip.value = [{ key: inputKey(), name: c.name, audioId: c.audioId, status: 'ready' }];
    transcript.value = c.transcript ?? '';
    voiceKind.value = 'clip';
  }
});
</script>

<template>
  <div class="speak">
    <div class="sections">
      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('speak.model') }}</h2>
        <ModelPicker v-model="model" />
        <div v-if="spec.data.value" class="caps">
          <Badge v-for="c in spec.data.value.capabilities.filter((x) => !x.startsWith('train'))" :key="c" :value="tOr(`capabilities.${c}`, c)" tone="neutral" />
        </div>
      </Surface>

      <Surface v-if="spec.data.value" :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('speak.voice') }}</h2>
        <SegmentedControl v-if="kindOptions.length > 1" v-model="voiceKind" :options="kindOptions" />
        <!-- Each way to give a voice is a pane; a change slides the way the control moved. -->
        <AxisPanes :value="voiceKind" :order="VOICE_KINDS">
          <div v-if="voiceKind === 'library'" key="library" class="pane">
            <VoicePicker v-model="voiceId" :model="model" />
          </div>
          <div v-else-if="voiceKind === 'clip'" key="clip" class="pane">
            <AudioSourceInput v-model="clip" :multiple="false" @added="onClipAdded" />
            <FieldRow>
              <TextField v-model="transcript" :label="t('speak.transcript')" :supporting-text="t('speak.transcriptHint')" />
              <template #end>
                <ModelGate :model="asrModel" compact>
                  <AppButton variant="tonal" :icon="icons.Languages" :disabled="!clip[0]?.audioId" :loading="transcribing" @click="transcribe">{{ t('speak.transcribe') }}</AppButton>
                </ModelGate>
              </template>
            </FieldRow>
          </div>
          <div v-else-if="voiceKind === 'preset'" key="preset" class="pane">
            <SelectField v-model="speaker" :label="t('speak.speaker')" :options="speakerOptions" />
          </div>
          <div v-else-if="voiceKind === 'design'" key="design" class="pane">
            <TextArea v-if="freeform" v-model="description" :label="t('speak.description')" :placeholder="t('speak.descriptionPlaceholder')" :rows="2" :max-length="1000" />
            <div v-if="designAttributes" class="attributes">
              <div v-for="(values, key) in designAttributes" :key="key" class="attribute">
                <span class="type-label-large">{{ key }}</span>
                <Chip
                  v-for="v in values"
                  :key="v"
                  :label="v"
                  :selected="attributes[key] === v"
                  @click="attributes = attributes[key] === v ? Object.fromEntries(Object.entries(attributes).filter(([k]) => k !== key)) : { ...attributes, [key]: v }"
                />
              </div>
            </div>
          </div>
          <div v-else key="auto" class="pane">
            <p class="type-body-medium muted">{{ t('speak.autoVoice') }}</p>
          </div>
        </AxisPanes>
      </Surface>

      <Surface v-if="spec.data.value" :level="0" shape="large" class="section">
        <ScriptEditor v-model="text" v-model:language="language" v-model:batch="batch" :model="model" :languages="spec.data.value.languages" :batchable="caps.has('batch')" @submit="run" />
      </Surface>

      <Surface v-if="spec.data.value && hasStyle" :level="0" shape="large" class="section">
        <h2 class="type-title-medium heading">{{ t('speak.style') }}</h2>
        <TextField v-if="caps.has('instruct')" v-model="instruct" :label="t('speak.instruct')" :placeholder="t('speak.instructPlaceholder')" />
        <EmotionPanel ref="emotionPanel" v-model="emotion" :emotions="spec.data.value.emotions" :vector="emotionWays.vector" :audio="emotionWays.audio" :text="emotionWays.text" />
      </Surface>

      <Surface v-if="spec.data.value" :level="0" shape="large" class="section">
        <div class="heading-row">
          <h2 class="type-title-medium heading">{{ t('params.title') }}</h2>
          <PresetMenu v-if="model" :model="model" :params="{ ...params, ...(seed !== null && takesSeed ? { seed } : {}) }" :options="options" @apply="applyPreset" />
        </div>
        <FieldRow v-if="takesSeed" :grow="false">
          <TextField
            :model-value="seed ?? ''"
            type="number"
            :label="t('params.seed')"
            :supporting-text="seed === null ? t('params.seedRandom') : t('params.seedLock')"
            class="seed"
            @update:model-value="seed = $event === '' || $event === null ? null : Number($event)"
          />
          <template #end>
            <IconButton :icon="icons.Dices" :label="t('params.seedRoll')" @click="rollSeed" />
            <IconButton v-if="seed !== null" :icon="icons.LockOpen" :label="t('params.seedRandom')" @click="seed = null" />
          </template>
        </FieldRow>
        <ParamsPanel v-model:params="params" v-model:options="options" :spec="spec.data.value" />
        <h3 class="type-title-small sub">{{ t('speak.output') }}</h3>
        <div class="grid">
          <SelectField :model-value="chosenFormat" :label="t('speak.format')" :options="formatOptions" @update:model-value="format = $event" />
          <SelectField v-model="sampleRate" :label="t('speak.sampleRate')" :options="rateOptions" />
          <SelectField :model-value="chosenNormalize" :label="t('speak.normalize')" :options="normalizeOptions" @update:model-value="normalize = $event" />
        </div>
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <ResultsList :outputs="outputs" @use-as-reference="useAsReference" @save-voice="saveVoice" @rerun="rerun" />
      </Surface>
    </div>

    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <StreamPlayerBar v-if="streamStatus && streamStatus.state !== 'failed'" :status="streamStatus" @stop="stopStreaming" />
      <JobsTray :jobs="sessionJobs" />
      <ModelGate :model="model">
        <div class="actions">
          <span v-if="!model" class="type-body-small muted status">{{ t('speak.chooseModel') }}</span>
          <span v-else-if="voiceProblem" class="type-body-small muted status">{{ voiceProblem }}</span>
          <span v-else-if="textProblem" class="type-body-small muted status">{{ textProblem }}</span>
          <Tooltip v-if="spec.data.value" :text="t(`speak.streamHints.${granularity}`)">
            <Switch v-model="stream" :label="t('speak.stream')" inline />
          </Tooltip>
          <AppButton variant="text" :disabled="!request" @click="showPlan">{{ t('speak.plan') }}</AppButton>
          <AppButton :icon="icons.Speech" :disabled="!canSpeak" :loading="busy" @click="run">{{ t('speak.speak') }}</AppButton>
        </div>
      </ModelGate>
    </PageFooter>
    <PlanDialog v-model:open="planOpen" :plan="plan" />
    <AppDialog v-model:open="keepOpen" :title="t('results.saveVoice')" width="small" :close-label="t('common.close')">
      <TextField v-model="keepName" :label="t('speak.saveVoiceName')" @enter="confirmKeep" />
      <template #actions>
        <AppButton variant="text" @click="keepOutput = null">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!keepName.trim()" @click="confirmKeep">{{ t('common.save') }}</AppButton>
      </template>
    </AppDialog>
  </div>
</template>

<style scoped>
/* One column of sections, each as wide as the page; the footer stays in view. */
.speak { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.heading { margin: 0; }
.heading-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--app-space-2); }
.sub { margin: var(--app-space-2) 0 0; }
.caps { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.pane { display: flex; flex-direction: column; gap: var(--app-space-4); min-width: 0; }
.seed { width: 220px; }
.attributes { display: flex; flex-direction: column; gap: var(--app-space-2); }
.attribute { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.attribute > span { min-width: 72px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); }
/* The status message takes the left; the stream switch, Plan and Speak keep together on the right. */
.actions > .status { margin-right: auto; }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
