<script setup lang="ts">
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useProjects } from '@/api/queries/projects';
import type { ProjectSummary } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import NewProjectDialog from '@/components/NewProjectDialog.vue';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { AppButton, AppCard, Badge, EmptyState, TRANSITIONS, icons, morphStyle, staggerStyle } from '@/ui';

/** Long-form projects: books, scripts and dialogues, each read line by line by a cast of voices. */
const { t } = useI18n();
const router = useRouter();
const projects = useProjects();
const newOpen = ref(false);

/** The share of lines in each state, for the card's bar. */
const shares = (p: ProjectSummary) => {
  const n = Math.max(1, p.stats.lines);
  return (['rendered', 'stale', 'failed', 'missing'] as const).map((k) => ({ state: k, width: `${(100 * p.stats[k]) / n}%` }));
};
</script>

<template>
  <div class="projects">
    <header class="bar">
      <h1 class="type-title-large title">{{ t('nav.projects') }}</h1>
      <AppButton :icon="icons.Plus" @click="newOpen = true">{{ t('projects.new') }}</AppButton>
    </header>
    <ErrorNotice v-if="projects.error.value" :error="projects.error.value" />
    <EmptyState v-if="projects.isSuccess.value && !projects.data.value?.length" :icon="icons.BookOpen" :title="t('projects.empty')" :text="t('projects.emptyHint')" />
    <TransitionGroup v-else :name="TRANSITIONS.list" tag="div" class="grid">
      <AppCard v-for="(p, i) in projects.data.value ?? []" :key="p.id" variant="filled" interactive class="card" :style="[staggerStyle(i), morphStyle(`project:${p.id}`)]" @activate="router.push(`/projects/${p.id}`)">
        <div class="head">
          <span class="type-title-medium name">{{ p.name }}</span>
          <Badge :value="`${p.stats.rendered}/${p.stats.lines}`" :tone="p.stats.failed ? 'error' : p.stats.rendered === p.stats.lines && p.stats.lines ? 'primary' : 'neutral'" />
        </div>
        <p v-if="p.description" class="type-body-small muted desc">{{ p.description }}</p>
        <div class="facts type-body-small muted">
          <span>{{ t('projects.chapters', { n: p.stats.chapters }) }} · {{ t('projects.lines', { n: p.stats.lines }) }}</span>
          <span>{{ t('projects.speakers', { n: p.speakers }) }}</span>
          <span v-if="p.stats.seconds">{{ formatDuration(p.stats.seconds) }}</span>
        </div>
        <div class="states" aria-hidden="true">
          <span v-for="s in shares(p)" :key="s.state" class="share" :class="s.state" :style="{ width: s.width }" />
        </div>
      </AppCard>
    </TransitionGroup>
    <NewProjectDialog v-model:open="newOpen" @created="(p) => router.push(`/projects/${p.id}`)" />
  </div>
</template>

<style scoped>
.projects { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.title { flex: 1; margin: 0; min-width: 0; }
.grid { position: relative; display: grid; grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr)); gap: var(--app-space-3); }
.card { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.desc { margin: 0; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.facts { display: flex; flex-wrap: wrap; gap: var(--app-space-1) var(--app-space-3); }
.states { display: flex; height: 4px; margin-top: auto; overflow: hidden; border-radius: 2px; background: var(--md-sys-color-surface-container-highest); }
.share.rendered { background: var(--md-sys-color-primary); }
.share.stale { background: var(--md-sys-color-tertiary); }
.share.failed { background: var(--md-sys-color-error); }
.share.missing { background: transparent; }
@container app-content (max-width: 599px) {
  .projects { padding: var(--app-space-3); }
}
</style>
