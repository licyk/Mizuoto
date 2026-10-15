import { reactive } from 'vue';
import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router';

declare module 'vue-router' {
  interface RouteMeta {
    /** The destination it belongs to, for the navigation and "started on another screen". */
    screen?: string;
  }
}

const SCREENS = ['speak', 'projects', 'tools', 'voices', 'train', 'models', 'history'];

/** The destination the app was last on (Preferences.lastScreen), opened when the address names none. */
function lastScreen(): string {
  try {
    const value = JSON.parse(localStorage.getItem('mizuoto:preferences') ?? '{}').lastScreen;
    return SCREENS.includes(value) ? value : 'speak';
  } catch {
    return 'speak';
  }
}

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: () => `/${lastScreen()}` },
  { path: '/speak', name: 'speak', component: () => import('@/views/SpeakView.vue'), meta: { screen: 'speak' } },
  { path: '/projects', name: 'projects', component: () => import('@/views/ProjectsView.vue'), meta: { screen: 'projects' } },
  { path: '/projects/:id', name: 'project', component: () => import('@/views/ProjectView.vue'), props: true, meta: { screen: 'projects' } },
  { path: '/tools', name: 'tools', component: () => import('@/views/ToolsView.vue'), meta: { screen: 'tools' } },
  { path: '/voices', name: 'voices', component: () => import('@/views/VoicesView.vue'), meta: { screen: 'voices' } },
  { path: '/voices/:id', name: 'voice', component: () => import('@/views/VoiceView.vue'), props: true, meta: { screen: 'voices' } },
  { path: '/train', name: 'train', component: () => import('@/views/TrainView.vue'), meta: { screen: 'train' } },
  { path: '/train/datasets/:id', name: 'dataset', component: () => import('@/views/DatasetView.vue'), props: true, meta: { screen: 'train' } },
  { path: '/train/experiments/:engine/:name', name: 'experiment', component: () => import('@/views/ExperimentView.vue'), props: true, meta: { screen: 'train' } },
  { path: '/models', name: 'models', component: () => import('@/views/ModelsView.vue'), meta: { screen: 'models' } },
  { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue'), meta: { screen: 'history' } },
  { path: '/jobs', name: 'jobs', component: () => import('@/views/JobsView.vue'), meta: { screen: 'jobs' } },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue'), meta: { screen: 'settings' } },
  { path: '/dev', name: 'dev', component: () => import('@/views/DevComponentsView.vue'), meta: { screen: 'dev' } },
  { path: '/:pathMatch(.*)*', redirect: '/speak' },
];

// Hash history: the UI works at any deployment sub-path with no server-side rewrites.
export const router = createRouter({ history: createWebHashHistory(), routes });

// The address each destination was last at, so the navigation goes back to the voice it left rather
// than to its list. Memory only.
const lastAddress = reactive<Record<string, string>>({});
router.afterEach((to) => {
  if (to.meta.screen) lastAddress[to.meta.screen] = to.fullPath;
});

export const addressOf = (screen: string): string => lastAddress[screen] ?? `/${screen}`;
