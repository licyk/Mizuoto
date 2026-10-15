<script setup lang="ts">
import { computed, onBeforeUnmount, onDeactivated, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { urls } from '@/api/client';
import { useOutputs } from '@/api/queries/outputs';
import { fetchLineRequest, projectUrls, useProject, useProjectActions } from '@/api/queries/projects';
import type { ChapterInput, LineInput, Project, ProjectInput, ProjectLine, SpeakerInput } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import JobCard from '@/components/JobCard.vue';
import ModelPicker from '@/components/ModelPicker.vue';
import ProjectSpeakerCard from '@/components/ProjectSpeakerCard.vue';
import { TRAIN_LANGUAGES } from '@/components/capabilities';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { useHandoffStore } from '@/stores/handoff';
import { useJobsStore } from '@/stores/jobs';
import { usePlayerStore } from '@/stores/player';
import {
  AppButton,
  AppDialog,
  AppIcon,
  AppMenu,
  AxisPanes,
  Badge,
  ConfirmDialog,
  IconButton,
  PageFooter,
  SegmentedButton,
  SelectField,
  Surface,
  Switch,
  Tabs,
  TextArea,
  TextField,
  TRANSITIONS,
  collapseHooks,
  icons,
  keepLeavingInPlace,
  morphStyle,
  useSnackbar,
  type MenuItem,
} from '@/ui';

/**
 * A long-form project: the script (chapters of lines, each with its speaker), the cast (a model and a
 * voice per speaker) and the export. Edits are kept in a draft and saved a moment after typing stops;
 * each line shows its take's state from the server (rendered, stale after an edit, failed, missing).
 */
const props = defineProps<{ id: string }>();
const { t } = useI18n();
const router = useRouter();
const snackbar = useSnackbar();
const player = usePlayerStore();
const jobs = useJobsStore();
const handoff = useHandoffStore();
const query = useProject(() => props.id);
const actions = useProjectActions();
const server = computed(() => query.data.value ?? null);
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

// -- the draft ----------------------------------------------------------------------------------
// ProjectInput with the fields the editor always fills made required.
type DraftLine = Omit<LineInput, 'text' | 'speaker' | 'ends_paragraph'> & { text: string; speaker: string | null; ends_paragraph: boolean };
type DraftChapter = Omit<ChapterInput, 'title' | 'lines'> & { title: string; lines: DraftLine[] };
type Draft = Omit<ProjectInput, 'speakers' | 'chapters'> & { speakers: SpeakerInput[]; chapters: DraftChapter[] };
const draft = ref<Draft | null>(null);
let edits = 0;
let saved = 0;
const dirty = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;

function toInput(p: Project): Draft {
  return {
    name: p.name,
    description: p.description,
    model: p.model,
    language: p.language,
    speakers: p.speakers.map((s) => ({ ...s, params: { ...s.params }, engine_options: { ...s.engine_options } })),
    chapters: p.chapters.map((c) => ({
      id: c.id,
      title: c.title ?? '',
      lines: c.lines.map((l) => ({ id: l.id, text: l.text, speaker: l.speaker ?? null, instruct: l.instruct, params: { ...l.params }, pause_ms: l.pause_ms, ends_paragraph: l.ends_paragraph, take: l.take })),
    })),
    line_pause_ms: p.line_pause_ms,
    paragraph_pause_ms: p.paragraph_pause_ms,
    chapter_pause_ms: p.chapter_pause_ms,
    format: p.format,
    sample_rate: p.sample_rate,
    normalize: p.normalize,
  };
}
watch(
  server,
  (p) => {
    if (p && !dirty.value) draft.value = toInput(p);
  },
  { immediate: true },
);

/** Something in the draft changed: save it shortly. */
function changed() {
  edits += 1;
  dirty.value = true;
  if (timer) clearTimeout(timer);
  timer = setTimeout(save, 800);
}
function save(): Promise<void> {
  if (timer) clearTimeout(timer);
  timer = null;
  if (!draft.value || !dirty.value) return Promise.resolve();
  const sent = edits;
  const body = draft.value;
  return new Promise((resolve) => {
    actions.save.mutate(
      { id: props.id, body },
      {
        onSuccess: (p) => {
          saved = sent;
          error.value = null;
          // Edits made while saving stay in the draft and are saved next.
          if (edits === sent) {
            dirty.value = false;
            draft.value = toInput(p);
          }
          resolve();
        },
        onError: (e) => {
          onError(e);
          resolve();
        },
      },
    );
  });
}
onDeactivated(() => void save());
onBeforeUnmount(() => void save());
const saveState = computed(() => (actions.save.isPending.value ? 'saving' : dirty.value ? 'unsaved' : saved ? 'saved' : ''));

const newId = (prefix: string) => `${prefix}${Date.now().toString(16)}${Math.random().toString(16).slice(2, 8)}`;

// -- tabs, chapters -------------------------------------------------------------------------------
const TABS = ['script', 'cast', 'export'] as const;
type Tab = (typeof TABS)[number];
const tab = ref<Tab>('script');
const tabs = computed(() => TABS.map((v) => ({ value: v, label: t(`projects.tabs.${v}`) })));
const chapterId = ref<string | null>(null);
const chapters = computed(() => draft.value?.chapters ?? []);
const chapter = computed<DraftChapter | null>(() => chapters.value.find((c) => c.id === chapterId.value) ?? chapters.value[0] ?? null);
watch(chapter, (c) => (chapterId.value = c?.id ?? null), { immediate: true });
const chapterTitle = (c: DraftChapter, i: number) => c.title || t('projects.untitled', { n: i + 1 });
// Chapters slide by their order: a later one in from the right.
const chapterOrder = computed(() => chapters.value.map((c) => c.id));

/** Each line's state, takes and error as the server last reported them. */
const serverLines = computed(() => new Map<string, ProjectLine>((server.value?.chapters ?? []).flatMap((c) => c.lines.map((l) => [l.id, l] as const))));
const stateOf = (l: DraftLine) => {
  const s = serverLines.value.get(l.id);
  if (!s) return l.text.trim() ? 'missing' : 'empty';
  if (s.text !== l.text && l.text.trim()) return 'stale';
  return s.state;
};
const counts = (c: DraftChapter) => {
  const done = c.lines.filter((l) => stateOf(l) === 'rendered').length;
  return `${done}/${c.lines.length}`;
};
function addChapter() {
  if (!draft.value) return;
  const c: DraftChapter = { id: newId('c'), title: '', lines: [{ id: newId('l'), text: '', speaker: null, ends_paragraph: false }] };
  draft.value.chapters.push(c);
  chapterId.value = c.id;
  changed();
}
function moveChapter(delta: number) {
  const list = draft.value?.chapters;
  const c = chapter.value;
  if (!list || !c) return;
  const i = list.indexOf(c);
  const j = i + delta;
  if (j < 0 || j >= list.length) return;
  list.splice(i, 1);
  list.splice(j, 0, c);
  changed();
}
const confirmChapter = ref(false);
function deleteChapter() {
  const list = draft.value?.chapters;
  const c = chapter.value;
  if (!list || !c) return;
  list.splice(list.indexOf(c), 1);
  chapterId.value = list[0]?.id ?? null;
  confirmChapter.value = false;
  changed();
}
const chapterMenu = computed<MenuItem[]>(() => [
  { id: 'up', label: t('projects.moveUp'), icon: icons.ArrowUp, disabled: chapters.value.indexOf(chapter.value as DraftChapter) <= 0 },
  { id: 'down', label: t('projects.moveDown'), icon: icons.ArrowDown, disabled: chapters.value.indexOf(chapter.value as DraftChapter) >= chapters.value.length - 1 },
  { id: 'delete', label: t('projects.deleteChapter'), icon: icons.Trash2, danger: true },
]);
function onChapterMenu(id: string) {
  if (id === 'up') moveChapter(-1);
  else if (id === 'down') moveChapter(1);
  else if (id === 'delete') confirmChapter.value = true;
}

// -- lines -----------------------------------------------------------------------------------------
const speakerOptions = computed(() => (draft.value?.speakers ?? []).map((s, i) => ({ value: i === 0 ? '' : s.id, label: s.name })));
const speakerValue = (l: DraftLine) => (l.speaker && l.speaker !== draft.value?.speakers[0]?.id ? l.speaker : '');
function setSpeaker(l: DraftLine, value: string | null) {
  l.speaker = value || null;
  changed();
}
function setText(l: DraftLine, text: string) {
  l.text = text;
  changed();
}
function addLine(after?: DraftLine) {
  const c = chapter.value;
  if (!c) return;
  const line: DraftLine = { id: newId('l'), text: '', speaker: after?.speaker ?? null, ends_paragraph: false };
  const at = after ? c.lines.indexOf(after) + 1 : c.lines.length;
  c.lines.splice(at, 0, line);
  changed();
}
const instructOpen = ref<Record<string, boolean>>({});
function lineMenu(l: DraftLine): MenuItem[] {
  const s = serverLines.value.get(l.id);
  const takes: MenuItem[] = (s?.takes ?? [])
    .slice()
    .reverse()
    .map((tk, i, all) => ({ id: `take:${tk.output_id}`, label: t('projects.take', { n: all.length - i, s: tk.duration.toFixed(1) }) + (tk.output_id === s?.take ? ' ✓' : ''), icon: icons.AudioLines }));
  const c = chapter.value;
  const i = c ? c.lines.indexOf(l) : -1;
  return [
    { id: 'insert', label: t('projects.insertBelow'), icon: icons.Plus },
    { id: 'paragraph', label: l.ends_paragraph ? t('projects.joinParagraph') : t('projects.endParagraph'), icon: icons.Split },
    ...(l.instruct || instructOpen.value[l.id] ? [] : [{ id: 'instruct', label: t('projects.addInstruct'), icon: icons.Sparkles }]),
    { id: 'up', label: t('projects.moveUp'), icon: icons.ArrowUp, disabled: i <= 0 },
    { id: 'down', label: t('projects.moveDown'), icon: icons.ArrowDown, disabled: !c || i >= c.lines.length - 1 },
    { id: 'speak', label: t('projects.tryInSpeak'), icon: icons.Speech },
    ...takes,
    { id: 'delete', label: t('common.delete'), icon: icons.Trash2, danger: true },
  ];
}
async function onLineMenu(l: DraftLine, id: string) {
  const c = chapter.value;
  if (!c) return;
  const i = c.lines.indexOf(l);
  if (id === 'insert') addLine(l);
  else if (id === 'paragraph') {
    l.ends_paragraph = !l.ends_paragraph;
    changed();
  } else if (id === 'instruct') instructOpen.value = { ...instructOpen.value, [l.id]: true };
  else if (id === 'up' || id === 'down') {
    const j = id === 'up' ? i - 1 : i + 1;
    if (j < 0 || j >= c.lines.length) return;
    c.lines.splice(i, 1);
    c.lines.splice(j, 0, l);
    changed();
  } else if (id === 'delete') {
    c.lines.splice(i, 1);
    changed();
  } else if (id === 'speak') {
    await save();
    try {
      handoff.sendRequest(await fetchLineRequest(props.id, l.id));
      router.push('/speak');
    } catch (e) {
      onError(e);
    }
  } else if (id.startsWith('take:')) {
    actions.useTake.mutate({ id: props.id, line: l.id, output: id.slice(5) }, { onError });
  }
}
/** Take a line's style instruction away (the line speaks in its speaker's style again). */
function removeInstruct(l: DraftLine) {
  instructOpen.value = { ...instructOpen.value, [l.id]: false };
  if (l.instruct) {
    l.instruct = null;
    changed();
  }
}
const takeKey = (l: DraftLine) => `project:${props.id}:${l.id}`;
function play(l: DraftLine) {
  const take = serverLines.value.get(l.id)?.take;
  if (take) player.play({ key: takeKey(l), url: urls.output(take), label: l.text });
}

// -- rendering ---------------------------------------------------------------------------------------
const renderJob = computed(() => jobs.active.find((j) => j.kind === 'project_render' && (j.request as Record<string, unknown>).project === server.value?.id) ?? null);
const exportJob = computed(() => jobs.active.find((j) => j.kind === 'project_export' && (j.request as Record<string, unknown>).project === server.value?.id) ?? null);
const todo = (lines: DraftLine[]) => lines.filter((l) => l.text.trim() && stateOf(l) !== 'rendered').length;
const chapterTodo = computed(() => (chapter.value ? todo(chapter.value.lines) : 0));
const allTodo = computed(() => chapters.value.reduce((n, c) => n + todo(c.lines), 0));
async function render(body: { chapters?: string[]; lines?: string[]; force?: boolean }) {
  await save();
  actions.render.mutate({ id: props.id, body }, { onError });
}

// -- cast ------------------------------------------------------------------------------------------
const linesOf = (s: SpeakerInput, i: number) => chapters.value.reduce((n, c) => n + c.lines.filter((l) => (i === 0 ? !l.speaker || l.speaker === s.id : l.speaker === s.id)).length, 0);
function addSpeaker() {
  if (!draft.value) return;
  const n = draft.value.speakers.length + 1;
  draft.value.speakers.push({ id: newId('s'), name: t('projects.newSpeaker', { n }) });
  changed();
}
const removing = ref<number | null>(null);
function removeSpeaker(i: number) {
  const d = draft.value;
  if (!d) return;
  const s = d.speakers[i];
  for (const c of d.chapters) for (const l of c.lines) if (l.speaker === s.id) l.speaker = null;
  d.speakers.splice(i, 1);
  removing.value = null;
  changed();
}
const projectModel = computed({
  get: () => draft.value?.model ?? null,
  set: (v) => {
    if (!draft.value) return;
    draft.value.model = v;
    changed();
  },
});

// -- import ----------------------------------------------------------------------------------------
const importOpen = ref(false);
const importText = ref('');
const importSplit = ref<'line' | 'sentence'>('line');
const importMode = ref<'append' | 'chapter' | 'replace'>('append');
const splitOptions = computed(() => (['line', 'sentence'] as const).map((v) => ({ value: v, label: t(`projects.split.${v}`) })));
const modeOptions = computed(() => (['append', 'chapter', 'replace'] as const).map((v) => ({ value: v, label: t(`projects.importModes.${v}`) })));
async function runImport() {
  await save();
  actions.importScript.mutate(
    { id: props.id, body: { text: importText.value, split: importSplit.value, replace: importMode.value === 'replace', chapter: importMode.value === 'chapter' ? chapter.value?.id : null } },
    {
      onSuccess: (p) => {
        importOpen.value = false;
        importText.value = '';
        draft.value = toInput(p);
        dirty.value = false;
      },
      onError,
    },
  );
}

// -- export ----------------------------------------------------------------------------------------
const exportsQuery = useOutputs(() => ({ kind: 'project' }));
const exports = computed(() => (exportsQuery.data.value?.items ?? []).filter((o) => (o.meta as Record<string, unknown>).project === server.value?.id));
const allowMissing = ref(false);
async function exportAudio(chapterIds: string[] | null) {
  await save();
  actions.exportAudio.mutate({ id: props.id, body: { chapters: chapterIds, allow_missing: allowMissing.value } }, { onError });
}
function setting<K extends keyof Draft>(key: K, value: Draft[K]) {
  if (!draft.value) return;
  draft.value[key] = value;
  changed();
}
const languageOptions = computed(() => [{ value: '', label: t('projects.languageAny') }, ...TRAIN_LANGUAGES.map((c) => ({ value: c, label: t(`languages.${c}`) }))]);
const formatOptions = ['wav', 'flac', 'mp3', 'ogg', 'm4a'].map((f) => ({ value: f, label: f.toUpperCase() }));
const normalizeOptions = computed(() => (['loudness', 'peak', 'none'] as const).map((v) => ({ value: v, label: t(`projects.normalizes.${v}`) })));
const playExport = (id: string, name: string) => player.play({ key: `export:${id}`, url: urls.output(id), label: name });

// -- delete ----------------------------------------------------------------------------------------
const confirmDelete = ref(false);
function remove() {
  const p = server.value;
  if (!p) return;
  if (timer) clearTimeout(timer);
  dirty.value = false;
  actions.remove.mutate(p.id, {
    onSuccess: () => {
      snackbar.show(t('projects.deleted', { name: p.name }));
      router.push('/projects');
    },
    onError,
  });
  confirmDelete.value = false;
}
</script>

<template>
  <div class="project" :style="morphStyle(`project:${id}`)">
    <div v-if="draft && server" class="sections">
      <header class="head">
        <IconButton :icon="icons.ArrowLeft" :label="t('common.back')" @click="router.push('/projects')" />
        <h1 class="type-headline-small title">{{ draft.name }}</h1>
        <span v-if="saveState" class="type-body-small muted">{{ t(`projects.${saveState}`) }}</span>
        <Badge :value="`${server.stats.rendered}/${server.stats.lines} · ${formatDuration(server.stats.seconds)}`" :tone="server.stats.failed ? 'error' : 'neutral'" />
      </header>
      <Tabs v-model="tab" :tabs="tabs" />

      <AxisPanes :value="tab" :order="TABS">
        <div v-if="tab === 'script'" key="script" class="editor">
          <Surface :level="0" shape="large" class="chapters">
            <h2 class="type-title-small">{{ t('projects.chapterList') }}</h2>
            <ol class="chapter-list">
              <li v-for="(c, i) in chapters" :key="c.id">
                <button type="button" class="chapter-item" :class="{ current: c.id === chapter?.id }" @click="chapterId = c.id">
                  <span class="type-body-medium chapter-name">{{ chapterTitle(c, i) }}</span>
                  <span class="type-label-small muted">{{ counts(c) }}</span>
                </button>
              </li>
            </ol>
            <AppButton variant="text" :icon="icons.Plus" @click="addChapter">{{ t('projects.addChapter') }}</AppButton>
          </Surface>

          <Surface :level="0" shape="large" class="lines">
            <AxisPanes v-if="chapter" :value="chapter.id" :order="chapterOrder">
              <div :key="chapter.id" class="chapter">
                <div class="chapter-head">
                  <TextField
                    :model-value="chapter.title"
                    :label="t('projects.chapterTitle')"
                    class="grow"
                    @update:model-value="
                      (chapter.title = String($event ?? '')),
                      changed()
                    "
                  />
                  <AppMenu :items="chapterMenu" @select="onChapterMenu">
                    <template #default="{ toggle }">
                      <IconButton :icon="icons.MoreVertical" :label="t('common.actions')" @click="toggle" />
                    </template>
                  </AppMenu>
                </div>
                <TransitionGroup :name="TRANSITIONS.list" tag="ol" class="line-list" v-bind="keepLeavingInPlace">
                  <li v-for="l in chapter.lines" :key="l.id" class="line-item" :class="[stateOf(l), { paragraph: l.ends_paragraph }]">
                    <div class="line-row">
                      <span class="dot" :class="stateOf(l)" :title="t(`projects.states.${stateOf(l)}`)" />
                      <SelectField :model-value="speakerValue(l)" :label="t('projects.speaker')" :options="speakerOptions" class="who" @update:model-value="setSpeaker(l, $event)" />
                      <TextArea :model-value="l.text" :rows="2" :show-count="false" :label="t('projects.lineText')" class="text" @update:model-value="setText(l, $event)" @submit="addLine(l)" />
                      <div class="line-actions">
                        <IconButton
                          :icon="player.isPlaying(takeKey(l)) ? icons.Pause : icons.Play"
                          :label="t('results.play')"
                          :disabled="!serverLines.get(l.id)?.take"
                          @click="play(l)"
                        />
                        <IconButton :icon="icons.RefreshCw" :label="t('projects.renderLine')" :disabled="!l.text.trim() || !!renderJob" @click="render({ lines: [l.id], force: true })" />
                        <AppMenu :items="lineMenu(l)" @select="onLineMenu(l, $event)">
                          <template #default="{ toggle }">
                            <IconButton :icon="icons.MoreVertical" :label="t('common.actions')" @click="toggle" />
                          </template>
                        </AppMenu>
                      </div>
                    </div>
                    <!-- The line's style instruction: in line with its text, removable; it opens and closes in place. -->
                    <Transition :name="TRANSITIONS.collapse" v-bind="collapseHooks">
                      <div v-if="l.instruct || instructOpen[l.id]" class="instruct-box">
                        <div class="instruct">
                          <AppIcon :icon="icons.Sparkles" :size="20" class="instruct-icon" />
                          <TextField
                            :model-value="l.instruct ?? ''"
                            :label="t('projects.lineInstruct')"
                            class="grow"
                            @update:model-value="
                              (l.instruct = String($event ?? '') || null),
                              changed()
                            "
                          />
                          <IconButton :icon="icons.X" :label="t('projects.removeInstruct')" @click="removeInstruct(l)" />
                        </div>
                      </div>
                    </Transition>
                    <p v-if="stateOf(l) === 'failed'" class="type-body-small line-error">{{ serverLines.get(l.id)?.error }}</p>
                  </li>
                </TransitionGroup>
                <div class="row-end">
                  <AppButton variant="text" :icon="icons.Plus" @click="addLine()">{{ t('projects.addLine') }}</AppButton>
                </div>
              </div>
            </AxisPanes>
            <div v-else class="none">
              <p class="type-body-medium muted">{{ t('projects.noChapters') }}</p>
              <AppButton :icon="icons.FileText" @click="importOpen = true">{{ t('projects.importScript') }}</AppButton>
            </div>
          </Surface>
        </div>

        <div v-else-if="tab === 'cast'" key="cast" class="cast">
          <Surface :level="0" shape="large" class="section">
            <h2 class="type-title-medium">{{ t('projects.defaults') }}</h2>
            <ModelPicker v-model="projectModel" :label="t('projects.model')" />
            <SelectField
              :model-value="draft.language ?? ''"
              :label="t('projects.language')"
              :options="languageOptions"
              @update:model-value="
                (draft.language = $event || null),
                changed()
              "
            />
            <TextField
              :model-value="draft.name"
              :label="t('projects.name')"
              @update:model-value="
                (draft.name = String($event ?? '')),
                changed()
              "
            />
            <TextField
              :model-value="draft.description ?? ''"
              :label="t('common.description')"
              @update:model-value="
                (draft.description = String($event ?? '')),
                changed()
              "
            />
          </Surface>
          <p class="type-body-medium muted">{{ t('projects.castHint') }}</p>
          <ProjectSpeakerCard
            v-for="(s, i) in draft.speakers"
            :key="s.id"
            v-model="draft.speakers[i]"
            :project-model="draft.model ?? null"
            :narrator="i === 0"
            :lines="linesOf(s, i)"
            @change="changed"
            @remove="removing = i"
          />
          <div class="row-end">
            <AppButton variant="tonal" :icon="icons.UserPlus" @click="addSpeaker">{{ t('projects.addSpeaker') }}</AppButton>
          </div>
        </div>

        <div v-else key="export" class="cast">
          <Surface :level="0" shape="large" class="section">
            <h2 class="type-title-medium">{{ t('projects.exportSettings') }}</h2>
            <div class="grid">
              <TextField :model-value="draft.line_pause_ms" type="number" :min="0" :max="60000" :label="t('projects.linePause')" @update:model-value="setting('line_pause_ms', Number($event) || 0)" />
              <TextField :model-value="draft.paragraph_pause_ms" type="number" :min="0" :max="60000" :label="t('projects.paragraphPause')" @update:model-value="setting('paragraph_pause_ms', Number($event) || 0)" />
              <TextField :model-value="draft.chapter_pause_ms" type="number" :min="0" :max="60000" :label="t('projects.chapterPause')" @update:model-value="setting('chapter_pause_ms', Number($event) || 0)" />
              <SelectField :model-value="draft.format ?? 'wav'" :label="t('projects.format')" :options="formatOptions" @update:model-value="setting('format', ($event ?? 'wav') as ProjectInput['format'])" />
              <TextField :model-value="draft.sample_rate" type="number" :min="8000" :max="192000" :label="t('projects.sampleRate')" @update:model-value="setting('sample_rate', Number($event) || 24000)" />
              <SelectField :model-value="draft.normalize ?? 'loudness'" :label="t('projects.normalize')" :supporting-text="t('projects.normalizeHint')" :options="normalizeOptions" @update:model-value="setting('normalize', ($event ?? 'loudness') as ProjectInput['normalize'])" />
            </div>
            <Switch v-model="allowMissing" :label="t('projects.allowMissing')" :supporting-text="t('projects.allowMissingHint')" />
            <div class="row-end">
              <AppButton variant="tonal" :icon="icons.Download" :disabled="!chapter || !!exportJob" @click="exportAudio(chapter ? [chapter.id] : null)">{{ t('projects.exportChapter') }}</AppButton>
              <AppButton :icon="icons.Download" :disabled="!!exportJob" :loading="actions.exportAudio.isPending.value" @click="exportAudio(null)">{{ t('projects.exportAll') }}</AppButton>
            </div>
            <JobCard v-if="exportJob" :job="exportJob" compact />
          </Surface>
          <Surface :level="0" shape="large" class="section">
            <h2 class="type-title-medium">{{ t('projects.exports') }}</h2>
            <p v-if="!exports.length" class="type-body-medium muted">{{ t('projects.noExports') }}</p>
            <ul class="exports">
              <li v-for="o in exports" :key="o.id" class="export">
                <IconButton :icon="player.isPlaying(`export:${o.id}`) ? icons.Pause : icons.Play" :label="t('results.play')" @click="playExport(o.id, o.name)" />
                <span class="type-body-medium grow name">{{ o.name }}</span>
                <span class="type-body-small muted">{{ formatDuration(o.duration) }} · {{ o.format.toUpperCase() }}</span>
                <a :href="urls.output(o.id, true)" class="link"><IconButton :icon="icons.Download" :label="t('common.download')" /></a>
                <a :href="projectUrls.subtitles(o.id)" :download="`${o.name}.srt`" class="link"><IconButton :icon="icons.Captions" :label="t('projects.subtitles')" /></a>
              </li>
            </ul>
          </Surface>
        </div>
      </AxisPanes>
    </div>
    <ErrorNotice v-else-if="query.error.value" :error="query.error.value" class="pad" />
    <PageFooter>
      <ErrorNotice v-if="error" :error="error" />
      <JobCard v-if="renderJob" :job="renderJob" compact />
      <div class="actions">
        <AppButton variant="text" :icon="icons.Trash2" :disabled="!server" @click="confirmDelete = true">{{ t('common.delete') }}</AppButton>
        <AppButton variant="text" :icon="icons.FileText" :disabled="!server" @click="importOpen = true">{{ t('projects.importScript') }}</AppButton>
        <span class="spacer" />
        <AppButton variant="tonal" :icon="icons.Play" :disabled="!chapterTodo || !!renderJob" @click="chapter && render({ chapters: [chapter.id] })">{{ t('projects.renderChapter', { n: chapterTodo }) }}</AppButton>
        <AppButton :icon="icons.AudioLines" :disabled="!allTodo || !!renderJob" :loading="actions.render.isPending.value" @click="render({})">{{ t('projects.renderAll', { n: allTodo }) }}</AppButton>
      </div>
    </PageFooter>

    <AppDialog v-model:open="importOpen" :title="t('projects.importScript')" width="large" :close-label="t('common.close')">
      <div class="dialog-body">
        <TextArea v-model="importText" :label="t('projects.script')" :supporting-text="t('projects.scriptHint')" :rows="12" />
        <div class="choice">
          <span class="type-body-medium">{{ t('projects.splitLabel') }}</span>
          <SegmentedButton v-model="importSplit" :options="splitOptions" />
        </div>
        <div class="choice">
          <span class="type-body-medium">{{ t('projects.importInto') }}</span>
          <SegmentedButton v-model="importMode" :options="modeOptions" />
        </div>
      </div>
      <template #actions>
        <AppButton variant="text" @click="importOpen = false">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!importText.trim()" :loading="actions.importScript.isPending.value" @click="runImport">{{ t('projects.import') }}</AppButton>
      </template>
    </AppDialog>
    <ConfirmDialog
      v-model:open="confirmDelete"
      :title="t('common.delete')"
      :message="t('projects.deleteConfirm', { name: server?.name ?? '' })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="remove"
    />
    <ConfirmDialog
      v-model:open="confirmChapter"
      :title="t('projects.deleteChapter')"
      :message="t('projects.deleteChapterConfirm', { name: chapter ? chapterTitle(chapter, chapters.indexOf(chapter)) : '', n: chapter?.lines.length ?? 0 })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="deleteChapter"
    />
    <ConfirmDialog
      :open="removing !== null"
      :title="t('projects.removeSpeaker')"
      :message="t('projects.removeSpeakerConfirm', { name: removing !== null ? (draft?.speakers[removing]?.name ?? '') : '' })"
      :confirm-label="t('common.remove')"
      :cancel-label="t('common.cancel')"
      danger
      @update:open="(o: boolean) => !o && (removing = null)"
      @confirm="removing !== null && removeSpeaker(removing)"
    />
  </div>
</template>

<style scoped>
.project { display: flex; flex-direction: column; min-height: 100%; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); flex: 1; padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.section { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); min-width: 0; }
.section h2, .section p { margin: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.title { margin: 0; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.editor { display: grid; grid-template-columns: minmax(180px, 240px) 1fr; gap: var(--app-space-4); align-items: start; min-width: 0; }
.chapters { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-3); position: sticky; top: var(--app-space-3); min-width: 0; }
.chapters h2 { margin: 0; padding: 0 var(--app-space-2); }
.chapter-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.chapter-item {
  display: flex; align-items: center; gap: var(--app-space-2); width: 100%; min-width: 0; padding: var(--app-space-2) var(--app-space-3);
  border: none; border-radius: var(--md-sys-shape-corner-full); background: transparent; color: var(--md-sys-color-on-surface); cursor: pointer; text-align: start;
}
.chapter-item:hover { background: var(--md-sys-color-surface-container-high); }
.chapter-item.current { background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.chapter-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lines { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); min-width: 0; }
.chapter-head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.line-list { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.line-item { display: flex; flex-direction: column; gap: var(--app-space-1); min-width: 0; }
.line-item.paragraph { padding-bottom: var(--app-space-3); border-bottom: 1px solid var(--md-sys-color-outline-variant); }
.line-row { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.who { flex: 0 0 160px; min-width: 0; }
.text { flex: 1; min-width: 0; }
.line-actions { display: flex; align-items: center; }
.dot { flex: 0 0 auto; width: 10px; height: 10px; border-radius: 50%; background: var(--md-sys-color-outline-variant); }
.dot.rendered { background: var(--md-sys-color-primary); }
.dot.stale { background: transparent; box-shadow: inset 0 0 0 2px var(--md-sys-color-tertiary); }
.dot.failed { background: var(--md-sys-color-error); }
.dot.rendering { background: var(--md-sys-color-primary-container); outline: 2px solid var(--md-sys-color-primary); }
.chapter { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
/* Under the line's text, the same width: past the state dot and the speaker. */
.instruct-box { min-width: 0; }
.instruct { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; margin-inline-start: calc(10px + var(--app-space-2) + 160px + var(--app-space-2)); }
.instruct-icon { flex: none; color: var(--md-sys-color-on-surface-variant); }
.line-error { margin: 0; margin-inline-start: calc(10px + var(--app-space-2)); color: var(--md-sys-color-error); }
.none { display: flex; flex-direction: column; align-items: flex-start; gap: var(--app-space-3); }
.cast { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr)); gap: var(--app-space-3); }
.grow { flex: 1; min-width: 0; }
.row-end { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--app-space-2); }
.exports { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.export { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.link { display: inline-flex; color: inherit; }
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.spacer { flex: 1; }
.dialog-body { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.choice { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.pad { margin: var(--app-space-4); }
@container app-content (max-width: 839px) {
  .editor { grid-template-columns: 1fr; }
  .chapters { position: static; }
}
@container app-content (max-width: 599px) {
  .sections { padding: var(--app-space-3); }
  .line-row { flex-wrap: wrap; }
  .who { flex-basis: 100%; }
  .instruct { margin-inline-start: 0; }
}
</style>
