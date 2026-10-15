<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useRouter } from 'vue-router';
import { api, unwrap } from '@/api/client';
import { useModels } from '@/api/queries/models';
import { fetchRerun, useOutputs } from '@/api/queries/outputs';
import { useVoiceActions } from '@/api/queries/voices';
import type { Output } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ResultsList from '@/components/ResultsList.vue';
import { useI18n } from '@/i18n';
import { useHandoffStore } from '@/stores/handoff';
import { AppButton, AppDialog, FilterChip, SearchField, SelectField, TextField, useSnackbar } from '@/ui';

/** Everything spoken: search the text, filter by model and favourites, run again in Speak. */
const { t } = useI18n();
const router = useRouter();
const handoff = useHandoffStore();
const voiceActions = useVoiceActions();
const snackbar = useSnackbar();
const models = useModels();
const q = ref('');
const settledQ = ref('');
let timer: ReturnType<typeof setTimeout> | undefined;
watch(q, (v) => {
  clearTimeout(timer);
  timer = setTimeout(() => (settledQ.value = v), 300);
});
const model = ref<string>('');
const favourite = ref(false);
const limit = ref(50);
const page = useOutputs(() => ({ q: settledQ.value || undefined, model: model.value || undefined, favourite: favourite.value || undefined }), true, limit);
const items = computed(() => page.data.value?.items ?? []);
const modelOptions = computed(() => [{ value: '', label: t('history.allModels') }, ...(models.data.value ?? []).map((m) => ({ value: m.id, label: m.title }))]);
const error = ref<unknown>(null);

async function rerun(o: Output) {
  try {
    handoff.sendRequest(await fetchRerun(o.id));
    router.push('/speak');
  } catch (e) {
    error.value = e;
  }
}
async function useAsReference(o: Output) {
  try {
    const f = await unwrap(api.POST('/api/v1/audio/files/from-output/{output_id}', { params: { path: { output_id: o.id } } }));
    handoff.sendReference(f.id, o.name, o.text);
    router.push('/speak');
  } catch (e) {
    error.value = e;
  }
}
const keeping = ref<Output | null>(null);
const keepName = ref('');
const keepOpen = computed({ get: () => !!keeping.value, set: (v) => (keeping.value = v ? keeping.value : null) });
function keep() {
  const o = keeping.value;
  if (!o || !keepName.value.trim()) return;
  voiceActions.fromOutput.mutate({ output_id: o.id, name: keepName.value.trim() }, { onSuccess: (v) => snackbar.show(t('voices.created', { name: v.name })), onError: (e) => (error.value = e) });
  keeping.value = null;
}
</script>

<template>
  <div class="history">
    <header class="bar">
      <SearchField v-model="q" :label="t('history.search')" :placeholder="t('history.search')" class="search" />
      <SelectField v-model="model" :label="t('speak.model')" :options="modelOptions" class="model" />
      <FilterChip v-model="favourite" :label="t('history.favourites')" />
    </header>
    <ErrorNotice v-if="error" :error="error" />
    <ResultsList :outputs="items" :empty-text="t('history.empty')" @rerun="rerun" @use-as-reference="useAsReference" @save-voice="(o) => ((keeping = o), (keepName = ''))" />
    <div v-if="page.data.value?.next_cursor" class="more">
      <AppButton variant="text" @click="limit += 50">{{ t('common.loadMore') }}</AppButton>
    </div>
    <AppDialog v-model:open="keepOpen" :title="t('results.saveVoice')" width="small" :close-label="t('common.close')">
      <TextField v-model="keepName" :label="t('speak.saveVoiceName')" @enter="keep" />
      <template #actions>
        <AppButton variant="text" @click="keeping = null">{{ t('common.cancel') }}</AppButton>
        <AppButton :disabled="!keepName.trim()" @click="keep">{{ t('common.save') }}</AppButton>
      </template>
    </AppDialog>
  </div>
</template>

<style scoped>
.history { display: flex; flex-direction: column; gap: var(--app-space-4); padding: var(--app-space-4) var(--app-space-6) var(--app-space-6); }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-3); }
.search { flex: 1; min-width: 220px; }
.model { min-width: 220px; }
.more { display: flex; justify-content: center; }
@container app-content (max-width: 599px) {
  .history { padding: var(--app-space-3); }
}
</style>
