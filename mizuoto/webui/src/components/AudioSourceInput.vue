<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import { ApiError, api, unwrap, urls } from '@/api/client';
import { useMeta } from '@/api/queries/app';
import { registerServerPaths, uploadAudio } from '@/api/queries/audio';
import type { AudioFile, Output } from '@/api/types';
import { GUIDE_SECONDS, Recorder, micProblem } from '@/audio/recorder';
import OutputsPickerDialog from '@/components/OutputsPickerDialog.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { inputKey, type InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppIcon, DropZone, IconButton, LevelMeter, ProgressBar, TRANSITIONS, icons, staggerStyle } from '@/ui';

/**
 * The one way to give audio: drop files, record from the microphone, pick server files (a browser on
 * the server's own machine, inside ``paths.browse_roots``), or choose earlier results. Each becomes an
 * audio file on the server.
 */
const props = withDefaults(defineProps<{ multiple?: boolean; uploads?: boolean; outputs?: boolean; record?: boolean; label?: string }>(), {
  multiple: true,
  uploads: true,
  outputs: true,
  record: true,
  label: '',
});
const items = defineModel<InputItem[]>({ default: () => [] });
const emit = defineEmits<{ added: [InputItem] }>();
const { t } = useI18n();
const meta = useMeta();
const player = usePlayerStore();
const serverOpen = ref(false);
const outputsOpen = ref(false);

function add(item: InputItem) {
  items.value = props.multiple ? [...items.value, item] : [item];
}
function patch(key: string, p: Partial<InputItem>) {
  items.value = items.value.map((i) => (i.key === key ? { ...i, ...p } : i));
  const done = items.value.find((i) => i.key === key);
  if (done && p.status === 'ready') emit('added', done);
}

async function resolve(key: string, name: string, get: () => Promise<AudioFile>) {
  add({ key, name, audioId: null, status: 'uploading', progress: 0 });
  try {
    const f = await get();
    patch(key, { status: 'ready', audioId: f.id, duration: f.duration });
  } catch (e) {
    patch(key, { status: 'failed', error: e instanceof ApiError ? e.message : String(e) });
  }
}

async function onFiles(files: File[]) {
  for (const file of props.multiple ? files : files.slice(0, 1)) {
    const key = inputKey();
    await resolve(key, file.name, () => uploadAudio(file, (loaded, total) => patch(key, { progress: total ? loaded / total : 0 })));
  }
}

async function onServer(selected: { path: string; name: string; dir: boolean }[]) {
  for (const s of selected.filter((x) => !x.dir)) await resolve(inputKey(), s.name, async () => (await registerServerPaths([s.path]))[0]);
}

async function onOutputs(outs: Output[]) {
  for (const o of outs) await resolve(inputKey(), o.name, () => unwrap(api.POST('/api/v1/audio/files/from-output/{output_id}', { params: { path: { output_id: o.id } } })));
}

const remove = (key: string) => (items.value = items.value.filter((i) => i.key !== key));
// -- recording ----------------------------------------------------------------------------
const problem = micProblem();
const recorder = ref<Recorder | null>(null);
const levels = ref({ rmsDb: -120, peakDb: -120, seconds: 0 });
const recordError = ref('');
const inGuide = computed(() => levels.value.seconds >= GUIDE_SECONDS.min && levels.value.seconds <= GUIDE_SECONDS.max);
async function startRecording() {
  recordError.value = '';
  const r = new Recorder();
  r.onLevels = (l) => (levels.value = l);
  try {
    await r.start();
    levels.value = { rmsDb: -120, peakDb: -120, seconds: 0 };
    recorder.value = r;
  } catch (e) {
    recordError.value = t('source.micFailed', { reason: e instanceof Error ? e.message : String(e) });
  }
}
async function stopRecording() {
  const file = recorder.value?.stop() ?? null;
  recorder.value = null;
  if (file) await onFiles([file]);
}
function cancelRecording() {
  recorder.value?.cancel();
  recorder.value = null;
}
onBeforeUnmount(cancelRecording);

function play(i: InputItem) {
  if (i.audioId) player.play({ key: i.key, url: urls.audioFile(i.audioId), label: i.name });
}
</script>

<template>
  <section class="source" :aria-label="label || t('source.title')">
    <DropZone v-if="uploads" :label="t('source.drop')" :hint="t('source.dropHint')" accept="audio/*,video/*,.wav,.flac,.mp3,.m4a,.ogg,.opus,.aac,.wma" :multiple="multiple" @files="onFiles" />
    <div v-if="recorder" class="recording" role="group" :aria-label="t('source.recording')">
      <AppIcon :icon="icons.Mic" :size="20" class="rec" />
      <span class="type-label-large time" :class="{ ok: inGuide }">{{ formatDuration(levels.seconds) }}</span>
      <LevelMeter :rms-db="levels.rmsDb" :peak-db="levels.peakDb" :label="t('source.level')" class="meter" />
      <AppButton :icon="icons.Square" @click="stopRecording">{{ t('source.recordStop') }}</AppButton>
      <AppButton variant="text" @click="cancelRecording">{{ t('source.recordCancel') }}</AppButton>
      <span class="type-body-small muted guide">{{ t('source.recordGuide', { min: GUIDE_SECONDS.min, max: GUIDE_SECONDS.max }) }}</span>
    </div>
    <div class="buttons">
      <AppButton v-if="record && !recorder" variant="tonal" :icon="icons.Mic" :disabled="!!problem" @click="startRecording">{{ t('source.record') }}</AppButton>
      <AppButton v-if="meta.data.value?.local" variant="tonal" :icon="icons.HardDrive" @click="serverOpen = true">{{ t('source.serverFiles') }}</AppButton>
      <AppButton v-if="outputs" variant="tonal" :icon="icons.History" @click="outputsOpen = true">{{ t('source.outputs') }}</AppButton>
      <AppButton v-if="items.length > 1" variant="text" @click="items = []">{{ t('source.clear') }}</AppButton>
    </div>
    <p v-if="record && problem" class="type-body-small muted note">{{ t(`source.mic.${problem}`) }}</p>
    <p v-if="recordError" class="type-body-small err note">{{ recordError }}</p>
    <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="items">
      <li v-for="(i, n) in items" :key="i.key" class="item" :class="i.status" :style="staggerStyle(n)">
        <IconButton v-if="i.audioId" :icon="player.isPlaying(i.key) ? icons.Pause : icons.Play" :label="player.isPlaying(i.key) ? t('results.pause') : t('results.play')" @click="play(i)" />
        <AppIcon v-else :icon="icons.Music" :size="20" class="lead" />
        <span class="text">
          <span class="type-body-large name">{{ i.name }}</span>
          <span v-if="i.status === 'failed'" class="type-body-small err">{{ t('source.uploadFailed', { name: i.name }) }}: {{ i.error }}</span>
          <ProgressBar v-else-if="i.status === 'uploading'" :value="i.progress ?? null" :label="t('source.uploading', { name: i.name })" />
        </span>
        <span v-if="i.duration" class="type-body-small muted">{{ formatDuration(i.duration) }}</span>
        <IconButton :icon="icons.X" :label="t('common.remove')" @click="remove(i.key)" />
      </li>
    </TransitionGroup>
    <ServerPathDialog v-model:open="serverOpen" @select="onServer" />
    <OutputsPickerDialog v-model:open="outputsOpen" @select="onOutputs" />
  </section>
</template>

<style scoped>
.source { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.buttons { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.items { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.item { display: flex; align-items: center; gap: var(--app-space-2); padding: var(--app-space-1) var(--app-space-1) var(--app-space-1) var(--app-space-2); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container); min-width: 0; }
.lead { margin: 0 var(--app-space-3); }
.text { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1; min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.err { color: var(--md-sys-color-error); overflow-wrap: anywhere; }
.note { margin: 0; }
.muted { color: var(--md-sys-color-on-surface-variant); }
.recording { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); padding: var(--app-space-2) var(--app-space-3); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container-high); }
.rec { color: var(--md-sys-color-error); }
.time { font-variant-numeric: tabular-nums; min-width: 48px; }
.time.ok { color: var(--md-sys-color-primary); }
.meter { flex: 1; min-width: 120px; }
.guide { flex-basis: 100%; }
</style>
