<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { downloadUrl } from '@/api/client';
import { useSettings } from '@/api/queries/app';
import { usePeaks } from '@/api/queries/audio';
import { useOutputs } from '@/api/queries/outputs';
import { speak } from '@/api/queries/tts';
import { useVoice, useVoiceActions, useVoiceFits, voiceUrls } from '@/api/queries/voices';
import type { VoiceReference } from '@/api/types';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import ModelGate from '@/components/ModelGate.vue';
import ResultsList from '@/components/ResultsList.vue';
import VoiceWeights from '@/components/VoiceWeights.vue';
import type { InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { useHandoffStore } from '@/stores/handoff';
import { isFinished, useJobsStore } from '@/stores/jobs';
import { usePlayerStore } from '@/stores/player';
import { AppButton, Badge, ConfirmDialog, IconButton, PageFooter, SelectField, Surface, TextField, TRANSITIONS, Waveform, icons, morphStyle, useSnackbar } from '@/ui';

/** One voice: its clips and transcripts, its details, which models can use it, a quick try, export and delete. */
const props = defineProps<{ id: string }>();
const { t, tOr } = useI18n();
const router = useRouter();
const voiceQuery = useVoice(() => props.id);
const fits = useVoiceFits(() => props.id);
const actions = useVoiceActions();
const player = usePlayerStore();
const handoff = useHandoffStore();
const jobs = useJobsStore();
const snackbar = useSnackbar();
const voice = computed(() => voiceQuery.data.value ?? null);
// Transcribing needs the speech recognition model: while it is missing, its download takes Re-identify's place.
const settings = useSettings();
const asrModel = computed(() => settings.data.value?.tts.asr_model ?? null);
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

// -- details --------------------------------------------------------------------------------
const name = ref('');
const language = ref('');
const description = ref('');
const tags = ref('');
watch(
  voice,
  (v) => {
    if (!v) return;
    name.value = v.name;
    language.value = v.language ?? '';
    description.value = v.description;
    tags.value = v.tags.join(', ');
  },
  { immediate: true },
);
const languageOptions = computed(() => [{ value: '', label: t('languages.any') }, ...['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'yue'].map((c) => ({ value: c, label: t(`languages.${c}`) }))]);
function saveDetails() {
  actions.update.mutate(
    {
      id: props.id,
      patch: {
        name: name.value,
        language: language.value,
        description: description.value,
        tags: tags.value.split(',').map((s) => s.trim()).filter(Boolean),
      },
    },
    { onSuccess: () => snackbar.show(t('settings.saved')), onError },
  );
}

// -- references -----------------------------------------------------------------------------
const transcripts = ref<Record<string, string>>({});
watch(voice, (v) => (transcripts.value = Object.fromEntries((v?.references ?? []).map((r) => [r.sha256, r.transcript ?? '']))), { immediate: true });
const refKey = (r: VoiceReference) => `${props.id}:${r.sha256}`;
const play = (r: VoiceReference) => player.play({ key: refKey(r), url: voiceUrls.reference(props.id, r.sha256), label: voice.value?.name ?? '' });
function saveTranscript(r: VoiceReference) {
  actions.updateReference.mutate({ id: props.id, sha: r.sha256, patch: { transcript: transcripts.value[r.sha256] ?? '' } }, { onSuccess: () => snackbar.show(t('voices.transcriptSaved')), onError });
}
const added = ref<InputItem[]>([]);
function onAdded(item: InputItem) {
  if (!item.audioId) return;
  actions.addReference.mutate({ id: props.id, body: { audio_id: item.audioId } }, { onSuccess: () => (added.value = []), onError });
}

// -- compatibility, try it --------------------------------------------------------------------
const usable = computed(() => (fits.data.value ?? []).filter((f) => f.usable));
const tryModel = ref<string | null>(null);
watch(usable, (u) => {
  if (!tryModel.value || !u.some((f) => f.model === tryModel.value)) tryModel.value = u[0]?.model ?? null;
});
const tryText = ref(t('voices.tryText'));
const tryJobs = ref<string[]>([]);
const activeTries = computed(() => tryJobs.value.map((id) => jobs.byId[id]).filter((j) => j && !isFinished(j.state)));
const tryOutputs = useOutputs(() => ({ voice_id: voice.value?.id }), () => !!voice.value, 10);
async function tryIt() {
  if (!tryModel.value || !voice.value) return;
  error.value = null;
  try {
    const job = await speak({ model: tryModel.value, text: tryText.value, voice: voice.value.id });
    jobs.upsert(job, 'voices');
    tryJobs.value = [...tryJobs.value, job.id];
  } catch (e) {
    error.value = e;
  }
}

// -- export, delete -------------------------------------------------------------------------
const confirmDelete = ref(false);
function remove() {
  const v = voice.value;
  if (!v) return;
  actions.remove.mutate(v.id, {
    onSuccess: () => {
      snackbar.show(t('voices.deleted', { name: v.name }));
      router.push('/voices');
    },
    onError,
  });
  confirmDelete.value = false;
}
function useInSpeak() {
  handoff.sendVoice(props.id);
  router.push('/speak');
}
</script>

<template>
  <div class="voice" :style="morphStyle(`voice:${id}`)">
    <div v-if="voice" class="sections">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/voices')" />
        <h1 class="type-headline-small title">{{ voice.name }}</h1>
        <Badge :value="tOr(`voices.kinds.${voice.kind}`, voice.kind)" tone="neutral" />
      </header>
      <p v-if="voice.design" class="type-body-medium muted">{{ t('voices.designedWith', { model: voice.design.model }) }}: {{ voice.design.description }}</p>
      <p v-if="voice.preset" class="type-body-medium muted">{{ t('voices.presetOf', { speaker: voice.preset.speaker, model: voice.preset.model }) }}</p>

      <Surface v-if="voice.kind !== 'preset'" :level="0" shape="large" class="section">
        <div class="section-head">
          <h2 class="type-title-medium">{{ t('voices.references') }}</h2>
          <ModelGate :model="asrModel" compact>
            <AppButton variant="text" :icon="icons.Languages" :loading="actions.retranscribe.isPending.value" @click="actions.retranscribe.mutate(id, { onError })">{{ t('voices.retranscribe') }}</AppButton>
          </ModelGate>
        </div>
        <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="refs">
          <li v-for="r in voice.references" :key="r.sha256" class="ref">
            <div class="ref-line">
              <IconButton :icon="player.isPlaying(refKey(r)) ? icons.Pause : icons.Play" :label="t('results.play')" @click="play(r)" />
              <RefWave :voice-id="id" :sha="r.sha256" :progress="player.state.track?.key === refKey(r) && player.state.duration ? player.state.time / player.state.duration : 0" />
              <span class="type-body-small muted dur">{{ formatDuration(r.duration) }} · {{ r.sample_rate / 1000 }} kHz</span>
              <Badge v-if="r.primary" :value="t('voices.primary')" tone="primary" />
              <IconButton v-else :icon="icons.CircleDot" :label="t('voices.makePrimary')" @click="actions.updateReference.mutate({ id, sha: r.sha256, patch: { primary: true } }, { onError })" />
              <IconButton :icon="icons.Trash2" :label="t('common.remove')" :disabled="voice.references.length < 2" @click="actions.removeReference.mutate({ id, sha: r.sha256 }, { onError })" />
            </div>
            <div class="ref-line">
              <TextField :model-value="transcripts[r.sha256] ?? ''" :label="t('speak.transcript')" class="grow" @update:model-value="transcripts = { ...transcripts, [r.sha256]: String($event ?? '') }" @enter="saveTranscript(r)" />
              <AppButton variant="tonal" :disabled="(transcripts[r.sha256] ?? '') === (r.transcript ?? '')" @click="saveTranscript(r)">{{ t('common.save') }}</AppButton>
            </div>
          </li>
        </TransitionGroup>
        <h3 class="type-title-small sub">{{ t('voices.addClip') }}</h3>
        <AudioSourceInput v-model="added" :multiple="false" @added="onAdded" />
      </Surface>

      <Surface v-if="voice.kind !== 'preset' && voice.kind !== 'designed'" :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('voices.weights') }}</h2>
        <VoiceWeights :voice="voice" />
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('voices.compatibility') }}</h2>
        <p v-if="fits.isSuccess.value && !fits.data.value?.length" class="type-body-medium muted">{{ t('voices.noInstalledModels') }}</p>
        <ul class="fits">
          <li v-for="f in fits.data.value ?? []" :key="f.model" class="fit">
            <span class="type-body-large grow">{{ f.model }}</span>
            <template v-if="f.usable">
              <Badge :value="tOr(`voiceModes.${f.mode}`, f.mode ?? '')" tone="primary" />
              <Badge v-if="f.weights" :value="t('voices.ownWeights')" tone="primary" />
              <Badge v-for="n in f.needs" :key="n" :value="tOr(`fit.${n}`, n)" tone="warning" />
              <AppButton v-if="f.mode === 'clone' || f.mode === 'clone_no_transcript'" variant="text" @click="actions.prepare.mutate({ id, model: f.model }, { onError })">{{ t('voices.prepare') }}</AppButton>
            </template>
            <span v-else class="type-body-small muted">{{ f.reason }}</span>
          </li>
        </ul>
      </Surface>

      <Surface v-if="usable.length" :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('voices.tryIt') }}</h2>
        <div class="grid">
          <SelectField v-model="tryModel" :label="t('speak.model')" :options="usable.map((f) => ({ value: f.model, label: f.model }))" />
          <TextField v-model="tryText" :label="t('speak.text')" @enter="tryIt" />
        </div>
        <div class="row-end">
          <AppButton variant="tonal" :icon="icons.Speech" :disabled="!tryModel || !tryText.trim()" @click="tryIt">{{ t('speak.speak') }}</AppButton>
        </div>
        <JobCard v-for="j in activeTries" :key="j!.id" :job="j!" compact />
        <ResultsList :outputs="tryOutputs.data.value?.items ?? []" />
      </Surface>

      <Surface :level="0" shape="large" class="section">
        <h2 class="type-title-medium">{{ t('voices.details') }}</h2>
        <div class="grid">
          <TextField v-model="name" :label="t('voices.name')" />
          <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" />
          <TextField v-model="tags" :label="t('voices.tags')" />
        </div>
        <TextField v-model="description" :label="t('common.description')" />
        <div class="row-end">
          <AppButton variant="tonal" :icon="icons.Save" :loading="actions.update.isPending.value" @click="saveDetails">{{ t('common.save') }}</AppButton>
        </div>
      </Surface>
    </div>
    <ErrorNotice v-else-if="voiceQuery.error.value" :error="voiceQuery.error.value" class="pad" />
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <div class="actions">
        <AppButton variant="text" :icon="icons.Trash2" :disabled="!voice" @click="confirmDelete = true">{{ t('common.delete') }}</AppButton>
        <AppButton variant="text" :icon="icons.Download" :disabled="!voice" @click="downloadUrl(voiceUrls.export(id))">{{ t('voices.export') }}</AppButton>
        <AppButton :icon="icons.Speech" :disabled="!voice" @click="useInSpeak">{{ t('voices.useInSpeak') }}</AppButton>
      </div>
    </PageFooter>
    <ConfirmDialog
      v-model:open="confirmDelete"
      :title="t('common.delete')"
      :message="t('voices.deleteConfirm', { name: voice?.name ?? '' })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="remove"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent, h } from 'vue';

/** A reference clip's waveform, from the server's peaks. */
const RefWave = defineComponent({
  props: { voiceId: { type: String, required: true }, sha: { type: String, required: true }, progress: { type: Number, default: 0 } },
  setup(p) {
    const peaks = usePeaks(() => ({ kind: 'reference', id: `${p.voiceId}/${p.sha}` }), 400);
    return () => h('div', { class: 'wave' }, [h(Waveform, { peaks: peaks.data.value?.data ?? null, progress: p.progress, height: 36 })]);
  },
});
</script>

<style scoped>
.voice { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.section { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.section h2, .section h3 { margin: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.section-head { display: flex; align-items: center; gap: var(--app-space-2); }
.section-head h2 { flex: 1; }
.refs { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-3); }
.ref { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-2) var(--app-space-3); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0; }
.ref-line { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.ref-line :deep(.wave) { flex: 1; min-width: 0; }
.dur { flex: none; font-variant-numeric: tabular-nums; }
.grow { flex: 1; min-width: 0; }
.sub { margin: var(--app-space-2) 0 0; }
.fits { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.fit { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); min-height: 40px; min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: center; }
.row-end { display: flex; justify-content: flex-end; }
.actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--app-space-2); }
.pad { margin: var(--app-space-4) var(--app-space-6); }
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .section { padding: var(--app-space-4); }
}
</style>
