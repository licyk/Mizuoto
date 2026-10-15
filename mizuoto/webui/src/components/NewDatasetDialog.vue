<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useDatasetActions } from '@/api/queries/train';
import type { Dataset } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { TRAIN_LANGUAGES } from '@/components/capabilities';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, SelectField, TextField } from '@/ui';

/** An empty dataset: its name, language and whose voice it is; clips are added on its page. */
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ created: [Dataset] }>();
const { t } = useI18n();
const actions = useDatasetActions();
const name = ref('');
const language = ref('');
const description = ref('');
const error = ref<unknown>(null);
watch(open, (o) => {
  if (!o) return;
  name.value = '';
  language.value = '';
  description.value = '';
  error.value = null;
});
const languageOptions = computed(() => [{ value: '', label: t('languages.any') }, ...TRAIN_LANGUAGES.map((c) => ({ value: c, label: t(`languages.${c}`) }))]);

function create() {
  error.value = null;
  actions.create.mutate(
    { name: name.value.trim(), language: language.value || null, description: description.value.trim() },
    {
      onSuccess: (d) => {
        open.value = false;
        emit('created', d);
      },
      onError: (e) => (error.value = e),
    },
  );
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('train.newDataset')" :close-label="t('common.close')">
    <div class="body">
      <TextField v-model="name" :label="t('train.datasetName')" @enter="name.trim() && create()" />
      <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" />
      <TextField v-model="description" :label="t('common.description')" />
      <ErrorNotice v-if="error" :error="error" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!name.trim()" :loading="actions.create.isPending.value" @click="create">{{ t('train.create') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.body { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
</style>
