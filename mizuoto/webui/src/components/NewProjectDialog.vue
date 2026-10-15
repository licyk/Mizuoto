<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useProjectActions } from '@/api/queries/projects';
import type { Project } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import ModelPicker from '@/components/ModelPicker.vue';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, SegmentedButton, TextArea, TextField } from '@/ui';

/** A new project: its name, the model its narrator reads with, and optionally a script to start from. */
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ created: [Project] }>();
const { t } = useI18n();
const actions = useProjectActions();
const name = ref('');
const model = ref<string | null>(null);
const script = ref('');
const split = ref<'line' | 'sentence'>('line');
const error = ref<unknown>(null);
watch(open, (o) => {
  if (!o) return;
  name.value = '';
  script.value = '';
  split.value = 'line';
  error.value = null;
});
const splitOptions = computed(() => (['line', 'sentence'] as const).map((v) => ({ value: v, label: t(`projects.split.${v}`) })));
const busy = computed(() => actions.create.isPending.value || actions.importScript.isPending.value);

function create() {
  error.value = null;
  actions.create.mutate(
    { name: name.value.trim(), model: model.value, speakers: [{ id: 'narrator', name: t('projects.narrator') }] },
    {
      onSuccess: (p) => {
        const done = (project: Project) => {
          open.value = false;
          emit('created', project);
        };
        if (!script.value.trim()) return done(p);
        actions.importScript.mutate({ id: p.id, body: { text: script.value, split: split.value } }, { onSuccess: done, onError: (e) => (error.value = e) });
      },
      onError: (e) => (error.value = e),
    },
  );
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('projects.new')" width="large" :close-label="t('common.close')">
    <div class="body">
      <TextField v-model="name" :label="t('projects.name')" />
      <ModelPicker v-model="model" :label="t('projects.model')" />
      <TextArea v-model="script" :label="t('projects.script')" :supporting-text="t('projects.scriptHint')" :rows="8" />
      <div class="split">
        <span class="type-body-medium">{{ t('projects.splitLabel') }}</span>
        <SegmentedButton v-model="split" :options="splitOptions" />
      </div>
      <ErrorNotice v-if="error" :error="error" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!name.trim()" :loading="busy" @click="create">{{ t('train.create') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.body { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.split { display: flex; flex-wrap: wrap; align-items: center; gap: var(--app-space-2) var(--app-space-3); }
</style>
