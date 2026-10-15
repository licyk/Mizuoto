<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query';
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { RouterView, useRoute, useRouter, type RouteLocationNormalized } from 'vue-router';
import { connectSocket, disconnectSocket } from '@/api/socket';
import AuthDialog from '@/components/AuthDialog.vue';
import JobsSheet from '@/components/JobsSheet.vue';
import LoadedModelsSheet from '@/components/LoadedModelsSheet.vue';
import { useRuntime } from '@/api/queries/runtime';
import { useI18n } from '@/i18n';
import { addressOf } from '@/router';
import { useJobsStore } from '@/stores/jobs';
import { usePreferencesStore } from '@/stores/preferences';
import { applyTheme, watchSystemTheme } from '@/theme/applyTheme';
import { AppShell, IconButton, TRANSITIONS, icons, installViewTransitions, useSnackbar, type NavItem } from '@/ui';

const { t } = useI18n();
const qc = useQueryClient();
const prefs = usePreferencesStore();
const jobs = useJobsStore();
const route = useRoute();
const router = useRouter();
const snackbar = useSnackbar();
const sheetOpen = ref(false);
const runtimeOpen = ref(false);
const runtime = useRuntime();
const loaded = computed(() => runtime.data.value?.models.length ?? 0);

/** One kept-alive instance per destination, and per experiment or voice for their detail screens. */
// A voice's, a dataset's, an experiment's and a project's page is one per item (each keeps its own state).
const viewKey = (path: string, name: unknown) => (path.startsWith('/voices/') || path.startsWith('/train/') || path.startsWith('/projects/') ? path : String(name));

// Pages are kept alive, so each keeps its state while another is shown; the shell's scroller is
// shared, so each page's offset in it is kept here and put back before the page is seen again
// (inside the view transition's update, so there is no jump). Memory only: a refresh resets all.
// Registered before the view transitions so the offset is in place when their update finishes.
const shell = ref<InstanceType<typeof AppShell> | null>(null);
const offsets = new Map<string, number>();
router.beforeEach((to, from) => {
  if (from.name && viewKey(to.fullPath, to.name) !== viewKey(from.fullPath, from.name)) offsets.set(viewKey(from.fullPath, from.name), shell.value?.content?.scrollTop ?? 0);
  return true;
});
function restoreOffset() {
  const content = shell.value?.content;
  if (content) content.scrollTop = offsets.get(viewKey(route.fullPath, route.name)) ?? 0;
}
router.afterEach((to, from) => {
  if (viewTransitions.value && viewKey(to.fullPath, to.name) !== viewKey(from.fullPath, from.name)) nextTick(restoreOffset);
});
// An item's card and its page morph into each other (rvc-next's container transform), opening and
// going back alike: a project, a voice, a dataset, an experiment.
const morphKey = (r: RouteLocationNormalized): string | null => {
  const p = r.params as Record<string, string>;
  if (r.name === 'project') return `project:${p.id}`;
  if (r.name === 'voice') return `voice:${p.id}`;
  if (r.name === 'dataset') return `dataset:${p.id}`;
  if (r.name === 'experiment') return `experiment:${p.engine}/${p.name}`;
  return null;
};
const LISTS: Record<string, string> = { project: 'projects', voice: 'voices', dataset: 'train', experiment: 'train' };
const viewTransitions = installViewTransitions(router, {
  morph: (to, from) => (LISTS[String(to.name)] === from.name ? morphKey(to) : LISTS[String(from.name)] === to.name ? morphKey(from) : null),
});

const themeOptions = () => ({ mode: prefs.prefs.theme, sourceColor: prefs.prefs.sourceColor, contrast: prefs.prefs.contrast });
watch(() => [prefs.prefs.theme, prefs.prefs.sourceColor, prefs.prefs.contrast], () => applyTheme(themeOptions()), { immediate: true });
watch(() => prefs.prefs.motion, (m) => (document.documentElement.dataset.motion = m), { immediate: true });
const stopTheme = watchSystemTheme(themeOptions);
const { locale } = useI18n();
watch(locale, (l) => (document.documentElement.lang = l), { immediate: true });
watch(
  () => route.meta.screen,
  (s) => {
    if (s && ['speak', 'projects', 'tools', 'voices', 'train', 'models', 'history'].includes(s)) prefs.prefs.lastScreen = s;
  },
);

// A job started on another screen that finishes or fails is announced; a failure links to its log.
const stopJobs = jobs.onFinished((job, origin) => {
  if (origin && origin === route.meta.screen) return;
  // A project's render announces itself; its lines' speak jobs do not (a book has hundreds).
  if (job.origin === 'project') return;
  if (job.state === 'completed') snackbar.show(t('jobs.finished', { title: job.title }));
  else if (job.state === 'failed') snackbar.show(t('jobs.failed', { title: job.title }), { error: true, actionLabel: t('jobs.openLog'), action: () => (sheetOpen.value = true), timeout: 8000 });
});

onMounted(() => {
  connectSocket(qc);
  prefs.loadFromServer();
});
onBeforeUnmount(() => {
  stopTheme();
  stopJobs();
  disconnectSocket();
});

const nav = computed<NavItem[]>(() => [
  { to: addressOf('speak'), label: t('nav.speak'), icon: icons.Speech },
  { to: addressOf('projects'), label: t('nav.projects'), icon: icons.BookOpen },
  { to: addressOf('tools'), label: t('nav.tools'), icon: icons.WandSparkles },
  { to: addressOf('voices'), label: t('nav.voices'), icon: icons.Users },
  { to: addressOf('train'), label: t('nav.train'), icon: icons.GraduationCap },
  { to: addressOf('models'), label: t('nav.models'), icon: icons.Library },
  { to: addressOf('history'), label: t('nav.history'), icon: icons.History },
]);
const footer = computed<NavItem[]>(() => [{ to: '/settings', label: t('nav.settings'), icon: icons.Settings }]);
// The top bar always names the app; the rail shows where you are.
const title = computed(() => t('app.title'));
// The theme, at the far right of the top bar, as Hanaikada's: light → dark → follow the system; the
// icon shows the current choice (the same three as Settings › Appearance).
const NEXT_THEME = { light: 'dark', dark: 'system', system: 'light' } as const;
const cycleTheme = () => (prefs.prefs.theme = NEXT_THEME[prefs.prefs.theme]);
const themeIcon = computed(() => ({ light: icons.Sun, dark: icons.Moon, system: icons.SunMoon })[prefs.prefs.theme]);
const activeCount = computed(() => jobs.active.length);
</script>

<template>
  <AppShell ref="shell" :items="nav" :footer="footer" :title="title">
    <template #rail-top>
      <IconButton :icon="icons.Mark" :label="t('app.title')" tonal />
    </template>
    <template #actions>
      <IconButton :icon="activeCount ? icons.Loader2 : icons.ListChecks" :spin="activeCount > 0" :label="activeCount ? t('jobs.chip', { n: activeCount }) : t('jobs.idle')" :badge="activeCount || null" @click="sheetOpen = true" />
      <IconButton :icon="icons.Cpu" :label="loaded ? t('runtime.chip', { n: loaded }) : t('runtime.title')" :badge="loaded || null" @click="runtimeOpen = true" />
      <IconButton :icon="themeIcon" :label="`${t('settings.theme')}: ${t(`settings.themes.${prefs.prefs.theme}`)}`" @click="cycleTheme" />
    </template>
    <RouterView v-slot="{ Component, route: r }">
      <!-- With view transitions the browser animates the change; the view must render at once. -->
      <KeepAlive v-if="viewTransitions" :max="12">
        <component :is="Component" :key="viewKey(r.fullPath, r.name)" />
      </KeepAlive>
      <Transition v-else :name="TRANSITIONS.fadeThrough" mode="out-in" @enter="restoreOffset">
        <KeepAlive :max="12">
          <component :is="Component" :key="viewKey(r.fullPath, r.name)" />
        </KeepAlive>
      </Transition>
    </RouterView>
  </AppShell>
  <JobsSheet v-model:open="sheetOpen" />
  <LoadedModelsSheet v-model:open="runtimeOpen" />
  <AuthDialog />
</template>
