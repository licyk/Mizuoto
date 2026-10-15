<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useBrowse } from '@/api/queries/audio';
import type { BrowseEntry } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { formatBytes } from '@/format';
import { AppButton, AppDialog, AppIcon, Checkbox, EmptyState, IconButton, icons } from '@/ui';

/** Pick audio files (or a folder) on the server, inside the folders it allows. */
const props = withDefaults(defineProps<{ folders?: boolean; files?: boolean; title?: string }>(), { folders: false, files: true, title: '' });
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ select: [{ path: string; name: string; dir: boolean }[]] }>();
const { t } = useI18n();
const path = ref('');
const chosen = ref<Set<string>>(new Set());
const listing = useBrowse(path, open, () => !props.files);

watch(open, (o) => {
  if (o) chosen.value = new Set();
});
const entries = computed(() => listing.data.value?.entries ?? []);
const atRoots = computed(() => !path.value);

function enter(e: BrowseEntry) {
  if (e.is_dir) {
    path.value = e.path;
    chosen.value = new Set();
  } else toggle(e);
}
function toggle(e: BrowseEntry) {
  const next = new Set(chosen.value);
  if (next.has(e.path)) next.delete(e.path);
  else next.add(e.path);
  chosen.value = next;
}
function up() {
  path.value = listing.data.value?.parent ?? '';
}
function addFiles() {
  emit('select', entries.value.filter((e) => chosen.value.has(e.path)).map((e) => ({ path: e.path, name: e.name, dir: false })));
  open.value = false;
}
function addFolder() {
  if (!path.value) return;
  emit('select', [{ path: listing.data.value?.path ?? path.value, name: (listing.data.value?.path ?? path.value).split(/[\\/]/).pop() ?? path.value, dir: true }]);
  open.value = false;
}
</script>

<template>
  <AppDialog v-model:open="open" :title="title || t('server.title')" width="medium" :close-label="t('common.close')">
    <div class="bar">
      <IconButton :icon="icons.FolderUp" :label="t('server.up')" :disabled="atRoots" @click="up" />
      <span class="type-body-medium path">{{ atRoots ? t('server.roots') : listing.data.value?.path }}</span>
    </div>
    <ErrorNotice v-if="listing.error.value" :error="listing.error.value" />
    <EmptyState v-else-if="listing.isSuccess.value && !entries.length" :icon="icons.FolderOpen" :title="t('server.empty')" />
    <ul v-else class="entries">
      <li v-for="e in entries" :key="e.path" class="entry">
        <Checkbox v-if="!e.is_dir && files" dense :model-value="chosen.has(e.path)" :label="e.name" @update:model-value="toggle(e)" />
        <button v-else type="button" class="dir state-layer" :disabled="!e.is_dir" @click="enter(e)">
          <AppIcon :icon="e.is_dir ? icons.Folder : icons.Music" :size="20" />
          <span class="type-body-large name">{{ e.name }}</span>
        </button>
        <span v-if="!e.is_dir" class="type-body-small muted">{{ formatBytes(e.size) }}</span>
      </li>
    </ul>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton v-if="props.folders" variant="tonal" :disabled="atRoots" @click="addFolder">{{ t('server.addFolder') }}</AppButton>
      <AppButton v-if="props.files" :disabled="!chosen.size" @click="addFiles">{{ t('server.add', { n: chosen.size }) }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.bar { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.path { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; direction: rtl; text-align: left; }
.entries { list-style: none; margin: 0; padding: 0; max-height: 50vh; overflow: auto; }
.entry { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.dir { display: flex; align-items: center; gap: var(--app-space-2); flex: 1; min-width: 0; padding: var(--app-space-2); border: 0; background: none; color: inherit; font: inherit; text-align: start; border-radius: var(--md-sys-shape-corner-small); cursor: pointer; }
.dir:disabled { cursor: default; }
.name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
