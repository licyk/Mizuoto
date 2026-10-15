<script setup lang="ts">
import { computed, ref } from 'vue';
import { useMeta, useSettings } from '@/api/queries/app';
import { useEngines } from '@/api/queries/engines';
import { useAssetActions, useAssets, useModelActions, useModels } from '@/api/queries/models';
import type { AssetInfo, ModelInfo } from '@/api/types';
import EngineCard from '@/components/EngineCard.vue';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ResourceRow, { type ResourceState } from '@/components/ResourceRow.vue';
import ServerPathDialog from '@/components/ServerPathDialog.vue';
import { sizeText } from '@/components/capabilities';
import { useI18n } from '@/i18n';
import { useJobsStore } from '@/stores/jobs';
import { AppButton, AxisPanes, Badge, ConfirmDialog, Surface, Tabs, icons, useSnackbar } from '@/ui';

/** Models (the catalogue by engine, with downloads and imports) and Engines (availability, licences, devices). */
const { t, tx } = useI18n();
const meta = useMeta();
const settings = useSettings();
const models = useModels();
const engines = useEngines();
const actions = useModelActions();
const jobs = useJobsStore();
const snackbar = useSnackbar();
type Tab = 'models' | 'engines' | 'assets';
const tabOrder = ['models', 'engines', 'assets'] as const;
const tab = ref<Tab>('models');
const tabs = computed(() => (['models', 'engines', 'assets'] as Tab[]).map((v) => ({ value: v, label: t(`models.tabs.${v}`) })));
const assets = useAssets();
const assetActions = useAssetActions();
const assetGroups = computed(() =>
  (engines.data.value ?? []).map((e) => ({ engine: e, assets: (assets.data.value ?? []).filter((a) => a.engine === e.id) })).filter((g) => g.assets.length),
);
const assetJob = (a: AssetInfo) =>
  jobs.active.find((j) => (j.kind === 'asset_download' && ((j.request as Record<string, unknown>).assets as string[] | undefined)?.includes(a.id)) || (j.kind === 'model_download' && a.used_by.some((m) => ((j.request as Record<string, unknown>).models as string[] | undefined)?.includes(m))))?.id ?? null;
const assetDetails = (a: AssetInfo) => [sizeText(a.size_mb, null, () => ''), tx(a.description), a.used_by.length ? t('models.usedBy', { models: a.used_by.join(', ') }) : ''].filter(Boolean).join(' · ');
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

// Speech engines first, then the helpers (speech recognition), the demo engine last.
const rank = (e: { id: string; internal: boolean }) => (e.id === 'dummy' ? 2 : e.internal ? 1 : 0);
const groups = computed(() =>
  (engines.data.value ?? [])
    .map((e) => ({ engine: e, models: (models.data.value ?? []).filter((m) => m.engine === e.id) }))
    .filter((g) => g.models.length)
    .sort((a, b) => rank(a.engine) - rank(b.engine)),
);
const orderedEngines = computed(() => [...(engines.data.value ?? [])].sort((a, b) => rank(a) - rank(b)));
const downloadJob = (m: ModelInfo) => jobs.active.find((j) => j.kind === 'model_download' && ((j.request as Record<string, unknown>).models as string[] | undefined)?.includes(m.id))?.id ?? null;
const details = (m: ModelInfo) => [m.variant, m.state === 'builtin' ? '' : sizeText(m.size_mb, m.vram_mb, (gb) => t('models.vram', { gb })), tx(m.description)].filter(Boolean).join(' · ');

function verify(m: ModelInfo) {
  actions.verify.mutate(m.id, {
    onSuccess: (r) => (r.ok ? snackbar.show(t('models.verified', { n: r.checked })) : snackbar.error(t('models.damaged', { files: r.bad.join(', ') }))),
    onError,
  });
}
const deleting = ref<ModelInfo | null>(null);
const confirmOpen = computed({ get: () => !!deleting.value, set: (v) => (deleting.value = v ? deleting.value : null) });
function remove() {
  if (deleting.value) actions.remove.mutate(deleting.value.id, { onError });
  deleting.value = null;
}
const importOpen = ref(false);
function onImport(selected: { path: string }[]) {
  const path = selected[0]?.path;
  if (path) actions.importFolder.mutate({ path }, { onSuccess: (m) => snackbar.show(t('models.imported', { id: m.id })), onError });
}
</script>

<template>
  <div class="models">
    <!-- The tabs span the row, as on Train; a tab's own action sits at its end. -->
    <header class="bar">
      <Tabs v-model="tab" :tabs="tabs" class="tabs" />
      <AppButton v-if="tab === 'models' && meta.data.value?.local" variant="tonal" :icon="icons.FolderInput" @click="importOpen = true">{{ t('models.import') }}</AppButton>
    </header>
    <ErrorNotice v-if="error" :error="error" />
    <AxisPanes :value="tab" :order="tabOrder">
      <div v-if="tab === 'models'" key="models" class="sections">
        <Surface v-for="g in groups" :key="g.engine.id" :level="0" shape="large" class="section">
          <div class="group-head">
            <h2 class="type-title-medium">{{ tx(g.engine.title) }}</h2>
            <Badge v-if="!g.engine.available" :value="t('engines.unavailable')" tone="warning" />
          </div>
          <p v-if="g.engine.remote" class="type-body-small muted">{{ t('remote.modelsThere', { host: g.engine.remote, engine: g.engine.id }) }}</p>
          <p v-else-if="!g.engine.available" class="type-body-small muted">{{ t('models.engineMissing', { engine: tx(g.engine.title), list: g.engine.missing.join(', ') }) }}</p>
          <ResourceRow
            v-for="m in g.models"
            :key="m.id"
            :title="m.title"
            :details="details(m)"
            :state="m.state as ResourceState"
            :job-id="downloadJob(m)"
            :verifiable="m.state !== 'builtin'"
            :deletable="m.state !== 'builtin'"
            :remote="m.remote"
            @download="actions.download.mutate([m.id], { onError })"
            @verify="verify(m)"
            @delete="deleting = m"
          />
        </Surface>
      </div>
      <div v-else-if="tab === 'assets'" key="assets" class="sections">
        <p class="type-body-medium muted">{{ t('models.assetsHint') }}</p>
        <Surface v-for="g in assetGroups" :key="g.engine.id" :level="0" shape="large" class="section">
          <div class="group-head">
            <h2 class="type-title-medium">{{ tx(g.engine.title) }}</h2>
          </div>
          <ResourceRow
            v-for="a in g.assets"
            :key="a.id"
            :title="tx(a.title)"
            :details="assetDetails(a)"
            :state="a.state as ResourceState"
            :job-id="assetJob(a)"
            deletable
            @download="assetActions.download.mutate([a.id], { onError })"
            @delete="assetActions.remove.mutate(a.id, { onError })"
          />
        </Surface>
      </div>
      <div v-else key="engines" class="sections">
        <EngineCard v-for="e in orderedEngines" :key="e.id" :engine="e" :settings="(settings.data.value?.engines?.[e.id] as Record<string, unknown> | undefined) ?? null" />
      </div>
    </AxisPanes>
    <ServerPathDialog v-model:open="importOpen" folders :files="false" :title="t('models.import')" @select="onImport" />
    <ConfirmDialog
      v-model:open="confirmOpen"
      :title="t('common.delete')"
      :message="t('models.deleteConfirm', { name: deleting?.title ?? '' })"
      :confirm-label="t('common.delete')"
      :cancel-label="t('common.cancel')"
      danger
      @confirm="remove"
    />
  </div>
</template>

<style scoped>
.models { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
.tabs { flex: 1; min-width: 240px; }
.sections { display: flex; flex-direction: column; gap: var(--app-space-4); }
.section { display: flex; flex-direction: column; gap: var(--app-space-2); padding: var(--app-space-4) var(--app-space-6) var(--app-space-4); min-width: 0; }
.group-head { display: flex; align-items: center; gap: var(--app-space-2); }
.group-head h2 { margin: 0; flex: 1; min-width: 0; }
p { margin: 0; }
@container app-content (max-width: 599px) {
  .models { padding: var(--app-space-3); }
  .section { padding: var(--app-space-3); }
}
</style>
