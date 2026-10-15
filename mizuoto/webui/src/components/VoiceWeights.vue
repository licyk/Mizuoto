<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query';
import { computed, ref, watch } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { useEngines } from '@/api/queries/engines';
import { keys } from '@/api/queries/keys';
import type { S, Voice } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppButton, Badge, IconButton, ProgressBar, SelectField, icons, useSnackbar } from '@/ui';

/**
 * A voice's own fine-tuned weights, per engine (GPT-SoVITS: a GPT .ckpt and a SoVITS .pth): what it
 * has, and adding a set by uploading each file. Models of the engine with the same version use them.
 */
const props = defineProps<{ voice: Voice }>();
const { t } = useI18n();
const qc = useQueryClient();
const snackbar = useSnackbar();
const engines = useEngines();

// Real engines first; the demo engine last.
const takers = computed(() => (engines.data.value ?? []).filter((e) => e.weights).sort((a, b) => Number(a.id === 'dummy') - Number(b.id === 'dummy')));
const titleOf = (id: string) => takers.value.find((e) => e.id === id)?.title ?? id;
const have = computed(() => Object.entries(props.voice.weights ?? {}));

const engine = ref<string | null>(null);
watch(takers, (list) => (engine.value ??= list[0]?.id ?? null), { immediate: true });
const roles = computed(() => Object.entries(takers.value.find((e) => e.id === engine.value)?.weights?.roles ?? {}));
const engineOptions = computed(() => takers.value.map((e) => ({ value: e.id, label: e.title })));
const files = ref<Record<string, File | null>>({});
watch(engine, () => (files.value = {}));
const progress = ref<number | null>(null);
const busy = ref(false);
const ready = computed(() => roles.value.length > 0 && roles.value.every(([role]) => files.value[role]));

function choose(role: string, suffix: string) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = suffix;
  input.onchange = () => (files.value = { ...files.value, [role]: input.files?.[0] ?? null });
  input.click();
}

async function upload() {
  if (!engine.value || !ready.value) return;
  busy.value = true;
  progress.value = 0;
  const list = roles.value.map(([role]) => [role, files.value[role]!] as const);
  const total = list.reduce((n, [, f]) => n + f.size, 0);
  let before = 0;
  try {
    for (const [role, file] of list) {
      await uploadRaw<S['WeightsUpload']>(`/api/v1/voices/${props.voice.id}/weights/${engine.value}/${role}`, { name: file.name }, file, (loaded) => (progress.value = (before + loaded) / total));
      before += file.size;
    }
    files.value = {};
    snackbar.show(t('voices.weightsAdded', { engine: titleOf(engine.value) }));
    await qc.invalidateQueries({ queryKey: keys.voices });
  } catch (e) {
    snackbar.error(e instanceof Error ? e.message : String(e));
  } finally {
    busy.value = false;
    progress.value = null;
  }
}

async function remove(id: string) {
  try {
    await unwrap(api.DELETE('/api/v1/voices/{voice_id}/weights/{engine}', { params: { path: { voice_id: props.voice.id, engine: id } } }));
    await qc.invalidateQueries({ queryKey: keys.voices });
  } catch (e) {
    snackbar.error(e instanceof Error ? e.message : String(e));
  }
}
</script>

<template>
  <div v-if="takers.length" class="weights">
    <ul v-if="have.length" class="list">
      <li v-for="[id, w] in have" :key="id" class="row">
        <span class="type-body-large grow">{{ titleOf(id) }}</span>
        <Badge v-if="w.version" :value="w.version" tone="primary" />
        <span class="type-body-small muted names">{{ Object.values(w.names).join(' · ') }}</span>
        <IconButton :icon="icons.Trash2" :label="t('common.remove')" @click="remove(id)" />
      </li>
    </ul>
    <p v-else class="type-body-medium muted">{{ t('voices.noWeights') }}</p>
    <div class="add">
      <SelectField v-if="engineOptions.length > 1" v-model="engine" :label="t('voices.weightsEngine')" :options="engineOptions" />
      <AppButton v-for="[role, suffix] in roles" :key="role" variant="tonal" :icon="files[role] ? icons.Check : icons.FolderOpen" @click="choose(role, suffix)">
        {{ files[role]?.name ?? t('voices.weightsChoose', { role, suffix }) }}
      </AppButton>
      <AppButton :icon="icons.Upload" :disabled="!ready" :loading="busy" @click="upload">{{ t('voices.weightsAdd') }}</AppButton>
    </div>
    <ProgressBar v-if="progress !== null" :value="progress" :label="t('voices.weightsUploading')" />
    <p class="type-body-small muted">{{ t('voices.weightsHint') }}</p>
  </div>
</template>

<style scoped>
.weights { display: flex; flex-direction: column; gap: var(--app-space-3); }
.list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.row { display: flex; align-items: center; gap: var(--app-space-2); }
.grow { flex: 1; min-width: 0; }
.names { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 50%; }
.add { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2); }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
</style>
