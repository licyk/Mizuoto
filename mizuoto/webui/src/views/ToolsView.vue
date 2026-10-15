<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { api, unwrap } from '@/api/client';
import { useSettings } from '@/api/queries/app';
import { transcribeAudio } from '@/api/queries/audio';
import { useEngines } from '@/api/queries/engines';
import { useModels } from '@/api/queries/models';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobsTray from '@/components/JobsTray.vue';
import ModelGate from '@/components/ModelGate.vue';
import ModelPicker from '@/components/ModelPicker.vue';
import ResultsList from '@/components/ResultsList.vue';
import VoicePicker from '@/components/VoicePicker.vue';
import type { InputItem } from '@/components/inputs';
import { useSessionOutputs } from '@/components/sessionOutputs';
import { useI18n } from '@/i18n';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { AppButton, AxisPanes, FieldRow, IconButton, PageFooter, SegmentedControl, Surface, Tabs, TextArea, TextField, TRANSITIONS, icons, keepLeavingInPlace } from '@/ui';

/**
 * What some models do besides speaking text: say what a recording says in another voice (voice
 * conversion, CosyVoice), and replace parts of a recording with new words in the same voice (speech
 * editing, F5-TTS).
 */
const TOOLS = ['convert', 'edit'] as const;
type Tool = (typeof TOOLS)[number];
const { t } = useI18n();
const jobs = useJobsStore();
const settings = useSettings();
// Transcribing needs the speech recognition model: offered for download in the button's place when missing.
const asrModel = computed(() => settings.data.value?.tts.asr_model ?? null);
const tool = ref<Tool>('convert');
const tabs = computed(() => [
  { value: 'convert' as Tool, label: t('tools.convert') },
  { value: 'edit' as Tool, label: t('tools.edit') },
]);
const models = ref<Record<Tool, string | null>>({ convert: null, edit: null });
const model = computed({ get: () => models.value[tool.value], set: (v: string | null) => (models.value = { ...models.value, [tool.value]: v }) });
// Nothing chosen yet: the first ready model that can do it (an installed one before the demo).
const allModels = useModels();
const engines = useEngines();
watch(
  [tool, () => allModels.data.value, () => engines.data.value],
  ([which, list, engineList]) => {
    if (models.value[which] || !list || !engineList) return;
    const ready = new Set(engineList.filter((e) => !e.internal && e.enabled && e.licence_accepted && e.available).map((e) => e.id));
    const capability = which === 'convert' ? 'voice_conversion' : 'speech_edit';
    const usable = list.filter((m) => ready.has(m.engine) && ['installed', 'builtin'].includes(m.state) && m.capabilities.includes(capability));
    const chosen = usable.find((m) => m.state === 'installed') ?? usable[0];
    if (chosen) models.value = { ...models.value, [which]: chosen.id };
  },
  { immediate: true },
);

const source = ref<InputItem[]>([]);
const sourceId = computed(() => source.value.find((s) => s.status === 'ready')?.audioId ?? null);

// -- convert ------------------------------------------------------------------------------------
const TARGETS = ['library', 'clip'] as const;
type Target = (typeof TARGETS)[number];
const target = ref<Target>('library');
const targetOptions = computed(() => [
  { value: 'library' as Target, label: t('speak.voiceKinds.library') },
  { value: 'clip' as Target, label: t('speak.voiceKinds.clip') },
]);
const voiceId = ref<string | null>(null);
const targetClip = ref<InputItem[]>([]);
const targetId = computed(() => targetClip.value.find((s) => s.status === 'ready')?.audioId ?? null);

// -- edit ---------------------------------------------------------------------------------------
interface Span {
  /** Keys the row, so removing one row animates that row and leaves the others' fields in place. */
  key: number;
  start: number | null;
  end: number | null;
  text: string;
  duration: number | null;
}
const transcript = ref('');
let spanKeys = 0;
const newSpan = (): Span => ({ key: ++spanKeys, start: null, end: null, text: '', duration: null });
const spans = ref<Span[]>([newSpan()]);
const transcribing = ref(false);
watch(sourceId, () => (transcript.value = ''));
async function transcribe() {
  if (!sourceId.value) return;
  transcribing.value = true;
  try {
    transcript.value = (await transcribeAudio(sourceId.value)).text;
  } catch (e) {
    error.value = e;
  } finally {
    transcribing.value = false;
  }
}
const num = (v: unknown) => (v === '' || v === null || v === undefined ? null : Number(v));
const validSpans = computed(() => spans.value.filter((s) => s.start !== null && s.end !== null && s.end > s.start));

// -- run ----------------------------------------------------------------------------------------
const problem = computed(() => {
  if (!model.value) return t('speak.chooseModel');
  if (!sourceId.value) return t('tools.noSource');
  if (tool.value === 'convert') return (target.value === 'library' ? voiceId.value : targetId.value) ? '' : t('speak.noVoice');
  if (!transcript.value.trim()) return t('tools.noTranscript');
  return validSpans.value.length ? '' : t('tools.noSpans');
});
const error = ref<unknown>(null);
const busy = ref(false);
const sessionJobs = computed(() => jobs.fromScreen('tools').filter((j) => !isFinished(j.state)));
const outputs = useSessionOutputs('tools');

async function run() {
  if (problem.value || !model.value || !sourceId.value || busy.value) return;
  error.value = null;
  busy.value = true;
  try {
    const job =
      tool.value === 'convert'
        ? await unwrap(
            api.POST('/api/v1/tts/convert', {
              body: {
                model: model.value,
                source: { audio_id: sourceId.value },
                voice: target.value === 'library' ? voiceId.value : null,
                voice_spec: target.value === 'clip' ? { references: [{ audio_id: targetId.value }] } : null,
              },
            }),
          )
        : await unwrap(
            api.POST('/api/v1/tts/edit', {
              body: {
                model: model.value,
                source: { audio_id: sourceId.value },
                transcript: transcript.value.trim(),
                edits: validSpans.value.map((s) => ({ start: s.start!, end: s.end!, text: s.text.trim(), duration: s.duration })),
              },
            }),
          );
    jobs.upsert(job, 'tools');
  } catch (e) {
    error.value = e;
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="tools">
    <div class="sections">
      <Tabs v-model="tool" :tabs="tabs" />
      <!-- Each tool is a pane: a change slides the way the tabs moved. -->
      <AxisPanes :value="tool" :order="TOOLS">
        <div :key="tool" class="pane">
          <Surface :level="0" shape="large" class="section">
            <h2 class="type-title-medium heading">{{ t('speak.model') }}</h2>
            <ModelPicker v-model="model" :capability="tool === 'convert' ? 'voice_conversion' : 'speech_edit'" />
            <p class="type-body-medium muted">{{ t(`tools.${tool}Hint`) }}</p>
          </Surface>

          <Surface :level="0" shape="large" class="section">
            <h2 class="type-title-medium heading">{{ t('tools.source') }}</h2>
            <AudioSourceInput v-model="source" :multiple="false" />
          </Surface>

          <Surface v-if="tool === 'convert'" :level="0" shape="large" class="section">
            <h2 class="type-title-medium heading">{{ t('tools.target') }}</h2>
            <SegmentedControl v-model="target" :options="targetOptions" />
            <AxisPanes :value="target" :order="TARGETS">
              <div v-if="target === 'library'" key="library" class="pane">
                <VoicePicker v-model="voiceId" :model="model" />
              </div>
              <div v-else key="clip" class="pane">
                <AudioSourceInput v-model="targetClip" :multiple="false" />
              </div>
            </AxisPanes>
          </Surface>

          <Surface v-else :level="0" shape="large" class="section">
            <h2 class="type-title-medium heading">{{ t('tools.changes') }}</h2>
            <FieldRow>
              <TextArea v-model="transcript" :label="t('tools.transcript')" :supporting-text="t('tools.transcriptHint')" :rows="3" />
              <template #end>
                <ModelGate :model="asrModel" compact>
                  <AppButton variant="tonal" :icon="icons.Languages" :disabled="!sourceId" :loading="transcribing" @click="transcribe">{{ t('speak.transcribe') }}</AppButton>
                </ModelGate>
              </template>
            </FieldRow>
            <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="spans" v-bind="keepLeavingInPlace">
              <li v-for="(s, n) in spans" :key="s.key" class="span">
                <TextField :model-value="s.start ?? ''" type="number" :label="t('tools.start')" class="time" @update:model-value="s.start = num($event)" />
                <TextField :model-value="s.end ?? ''" type="number" :label="t('tools.end')" class="time" @update:model-value="s.end = num($event)" />
                <TextField v-model="s.text" :label="t('tools.newWords')" class="grow" />
                <TextField :model-value="s.duration ?? ''" type="number" :label="t('tools.duration')" :placeholder="t('common.auto')" class="time" @update:model-value="s.duration = num($event)" />
                <IconButton :icon="icons.X" :label="t('common.remove')" :disabled="spans.length < 2" @click="spans.splice(n, 1)" />
              </li>
            </TransitionGroup>
            <div>
              <AppButton variant="text" :icon="icons.Plus" @click="spans.push(newSpan())">{{ t('tools.addSpan') }}</AppButton>
            </div>
          </Surface>
        </div>
      </AxisPanes>

      <Surface :level="0" shape="large" class="section">
        <ResultsList :outputs="outputs" />
      </Surface>
    </div>

    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <JobsTray :jobs="sessionJobs" />
      <ModelGate :model="model">
        <div class="actions">
          <span v-if="problem" class="type-body-small muted">{{ problem }}</span>
          <AppButton :icon="tool === 'convert' ? icons.ArrowLeftRight : icons.Scissors" :disabled="!!problem" :loading="busy" @click="run">{{ t(`tools.${tool}Run`) }}</AppButton>
        </div>
      </ModelGate>
    </PageFooter>
  </div>
</template>

<style scoped>
.tools { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.heading { margin: 0; }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
.pane { display: flex; flex-direction: column; gap: var(--app-space-4); min-width: 0; }
.grow { flex: 1; min-width: 200px; }
.spans { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.span { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.time { width: 120px; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: var(--app-space-2); }
.actions > span { margin-right: auto; }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
