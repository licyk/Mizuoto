<script setup lang="ts">
import { computed } from 'vue';
import { useMeta } from '@/api/queries/app';
import { useEngines } from '@/api/queries/engines';
import { revealOutput, useDeleteOutputs, useUpdateOutput } from '@/api/queries/outputs';
import { downloadOutputsZip, downloadUrl, urls } from '@/api/client';
import type { Output } from '@/api/types';
import type { Track } from '@/audio/player';
import ResultWave from '@/components/ResultWave.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppMenu, Badge, EmptyState, IconButton, TRANSITIONS, icons, staggerStyle, useSnackbar, type MenuItem } from '@/ui';

/**
 * Synthesized results: a waveform, play, the text, the model and how long it took, and a menu:
 * download, reveal (on the server's machine), use as a reference, keep as a voice, run again, delete.
 */
const props = withDefaults(defineProps<{ outputs: Output[]; emptyText?: string; title?: string }>(), { emptyText: '', title: '' });
const emit = defineEmits<{ useAsReference: [Output]; saveVoice: [Output]; rerun: [Output] }>();
const { t } = useI18n();
const meta = useMeta();
const player = usePlayerStore();
const remove = useDeleteOutputs();
const update = useUpdateOutput();
const snackbar = useSnackbar();

const trackOf = (o: Output): Track => ({ key: o.id, url: urls.output(o.id), label: o.name });
const progressOf = (o: Output) => (player.state.track?.key === o.id && player.state.duration ? player.state.time / player.state.duration : 0);
function seek(o: Output, fraction: number) {
  if (player.state.track?.key !== o.id) player.play(trackOf(o)).then(() => player.seek(fraction * (o.duration ?? 0)));
  else player.seek(fraction * (player.state.duration || o.duration || 0));
}
const timings = (o: Output) => (o.meta?.timings ?? {}) as { total?: number; rtf?: number | null; load?: number };
const warnings = (o: Output) => ((o.meta?.warnings ?? []) as string[]).filter(Boolean);
// Licences that ask results to say where they come from ("Built with Fish Audio").
const engines = useEngines();
const notices = computed(() => new Map((engines.data.value ?? []).filter((e) => e.licence.notice).map((e) => [e.licence.id, e.licence.notice])));
const noticesOf = (o: Output) => [...new Set((o.licence_tags ?? []).map((tag) => notices.value.get(tag)).filter((n): n is string => !!n))];

const menu = (o: Output): MenuItem[] => [
  { id: 'favourite', label: o.favourite ? t('history.unfavourite') : t('history.favourite'), icon: icons.Tag },
  { id: 'download', label: t('common.download'), icon: icons.Download },
  ...(meta.data.value?.local ? [{ id: 'reveal', label: t('common.reveal'), icon: icons.FolderOpen }] : []),
  { id: 'reference', label: t('results.useAsReference'), icon: icons.ArrowRight },
  { id: 'voice', label: t('results.saveVoice'), icon: icons.UserPlus },
  { id: 'rerun', label: t('results.rerun'), icon: icons.Repeat },
  { id: 'delete', label: t('common.delete'), icon: icons.Trash2, danger: true },
];
function onMenu(o: Output, id: string) {
  if (id === 'favourite') update.mutate({ id: o.id, favourite: !o.favourite });
  else if (id === 'download') downloadUrl(urls.output(o.id, true));
  else if (id === 'reveal') revealOutput(o.id).catch((e) => snackbar.error(e.message));
  else if (id === 'reference') emit('useAsReference', o);
  else if (id === 'voice') emit('saveVoice', o);
  else if (id === 'rerun') emit('rerun', o);
  else if (id === 'delete') {
    if (player.state.track?.key === o.id) player.stop();
    remove.mutate([o.id], { onSuccess: () => snackbar.show(t('results.deleted')) });
  }
}
function downloadAll() {
  downloadOutputsZip(props.outputs.map((o) => o.id)).catch((e) => snackbar.error(e.message));
}
</script>

<template>
  <section class="results" :aria-label="title || t('results.title')">
    <header class="head">
      <h2 class="type-title-medium title">{{ title || t('results.title') }}</h2>
      <AppButton v-if="outputs.length > 1" variant="text" :icon="icons.Download" @click="downloadAll">{{ t('results.downloadAll') }}</AppButton>
    </header>
    <EmptyState v-if="!outputs.length" :icon="icons.AudioWaveform" :title="emptyText || t('results.empty')" />
    <TransitionGroup :name="TRANSITIONS.list" tag="ul" class="list" aria-live="polite">
      <li v-for="(o, i) in outputs" :key="o.id" class="row" :style="staggerStyle(i)">
        <IconButton :icon="player.isPlaying(o.id) ? icons.Pause : icons.Play" :label="player.isPlaying(o.id) ? t('results.pause') : t('results.play')" @click="player.play(trackOf(o))" />
        <div class="main">
          <div class="line">
            <span class="type-body-large name" :title="o.text ?? o.name">{{ o.text || o.name }}</span>
            <span class="type-body-small muted dur">{{ o.duration ? formatDuration(o.duration) : '' }}</span>
          </div>
          <ResultWave :output="o" :progress="progressOf(o)" @seek="seek(o, $event)" />
          <div class="facts type-body-small muted">
            <Badge v-if="o.favourite" :value="t('history.favourite')" tone="primary" />
            <Badge :value="o.model ?? ''" tone="neutral" />
            <span v-if="timings(o).total">{{ t('results.took', { s: timings(o).total!.toFixed(1) }) }}</span>
            <span v-if="timings(o).rtf">RTF {{ timings(o).rtf!.toFixed(2) }}</span>
            <span v-if="o.params?.seed !== undefined">{{ t('results.seed', { seed: String(o.params.seed) }) }}</span>
          </div>
          <p v-for="w in warnings(o)" :key="w" class="type-body-small warn">{{ w }}</p>
          <p v-for="n in noticesOf(o)" :key="n" class="type-label-medium notice">{{ n }}</p>
        </div>
        <AppMenu :items="menu(o)" @select="onMenu(o, $event)">
          <template #default="{ toggle }"><IconButton :icon="icons.MoreVertical" :label="t('common.actions')" @click="toggle" /></template>
        </AppMenu>
      </li>
    </TransitionGroup>
  </section>
</template>

<style scoped>
.results { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); }
.title { flex: 1; margin: 0; }
.list { position: relative; list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-2); }
.row { display: flex; align-items: flex-start; gap: var(--app-space-2); padding: var(--app-space-2); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-surface-container); min-width: 0; }
.main { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1; min-width: 0; }
.line { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dur { margin-left: auto; flex: none; font-variant-numeric: tabular-nums; }
.facts { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.warn { margin: 0; color: var(--md-sys-color-tertiary); }
.notice { margin: 0; color: var(--md-sys-color-on-surface-variant); }
</style>
