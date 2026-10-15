<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { importVoice, useVoices, voiceUrls } from '@/api/queries/voices';
import type { VoiceSummary } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import NewVoiceDialog from '@/components/NewVoiceDialog.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { useHandoffStore } from '@/stores/handoff';
import { usePlayerStore } from '@/stores/player';
import { AppButton, AppCard, Badge, EmptyState, FilterChip, IconButton, SearchField, TRANSITIONS, icons, morphStyle, staggerStyle, useSnackbar } from '@/ui';

/** The voice library: a grid of voices with search and kind filters, New voice and Import. */
const { t, tOr } = useI18n();
const router = useRouter();
const player = usePlayerStore();
const handoff = useHandoffStore();
const snackbar = useSnackbar();
const q = ref('');
const kind = ref<string | null>(null);
const voices = useVoices(() => ({ q: q.value || undefined, kind: kind.value ?? undefined }));
const list = computed(() => voices.data.value ?? []);
const newOpen = ref(false);
const error = ref<unknown>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const kinds = ['reference', 'designed', 'preset'];

function play(v: VoiceSummary) {
  if (v.primary) player.play({ key: `voice:${v.id}`, url: voiceUrls.reference(v.id, v.primary), label: v.name });
}
function useInSpeak(v: VoiceSummary) {
  handoff.sendVoice(v.id);
  router.push('/speak');
}
async function onImport(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  error.value = null;
  try {
    const v = await importVoice(file);
    snackbar.show(t('voices.imported', { name: v.name }));
    voices.refetch();
  } catch (e) {
    error.value = e;
  } finally {
    if (fileInput.value) fileInput.value.value = '';
  }
}
</script>

<template>
  <div class="voices">
    <header class="bar">
      <SearchField v-model="q" :label="t('voices.search')" :placeholder="t('voices.search')" class="search" />
      <div class="filters">
        <FilterChip v-for="k in kinds" :key="k" :label="t(`voices.kinds.${k}`)" :model-value="kind === k" @update:model-value="kind = $event ? k : kind === k ? null : kind" />
      </div>
      <AppButton variant="text" :icon="icons.Upload" @click="fileInput?.click()">{{ t('voices.import') }}</AppButton>
      <AppButton :icon="icons.UserPlus" @click="newOpen = true">{{ t('voices.new') }}</AppButton>
      <input ref="fileInput" type="file" accept=".zip,application/zip" hidden @change="onImport" />
    </header>
    <ErrorNotice v-if="error" :error="error" />
    <EmptyState v-if="voices.isSuccess.value && !list.length" :icon="icons.Users" :title="t('voices.empty')" :text="t('voices.emptyHint')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
      <AppCard v-for="(v, i) in list" :key="v.id" variant="filled" interactive class="card" :style="[staggerStyle(i), morphStyle(`voice:${v.id}`)]" @activate="router.push(`/voices/${v.id}`)">
        <div class="head">
          <span class="type-title-medium name">{{ v.name }}</span>
          <Badge :value="tOr(`voices.kinds.${v.kind}`, v.kind)" tone="neutral" />
        </div>
        <p v-if="v.description" class="type-body-small muted desc">{{ v.description }}</p>
        <div class="facts type-body-small muted">
          <span v-if="v.language">{{ tOr(`languages.${v.language}`, v.language) }}</span>
          <span v-if="v.references">{{ t('voices.clips', { n: v.references }) }} · {{ formatDuration(v.duration) }}</span>
          <span v-if="v.preset_model">{{ v.preset_model }}</span>
        </div>
        <div class="actions" @click.stop>
          <IconButton v-if="v.primary" :icon="player.isPlaying(`voice:${v.id}`) ? icons.Pause : icons.Play" :label="t('voices.play')" @click="play(v)" />
          <AppButton variant="text" :icon="icons.Speech" @click="useInSpeak(v)">{{ t('voices.useInSpeak') }}</AppButton>
        </div>
      </AppCard>
    </TransitionGroup>
    <NewVoiceDialog v-model:open="newOpen" @created="(v) => (snackbar.show(t('voices.created', { name: v.name })), router.push(`/voices/${v.id}`))" />
  </div>
</template>

<style scoped>
.voices { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.search { flex: 1; min-width: 220px; }
.filters { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.grid { position: relative; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3); }
.card { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.desc { margin: 0; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.facts { display: flex; flex-wrap: wrap; gap: var(--app-space-2); }
.actions { display: flex; align-items: center; gap: var(--app-space-1); margin-top: auto; }
@container app-content (max-width: 599px) {
  .voices { padding: var(--app-space-3); }
}
</style>
