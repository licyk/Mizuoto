<script setup lang="ts">
import { computed, ref } from 'vue';
import { usePresetActions, usePresets } from '@/api/queries/presets';
import type { Preset } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, AppMenu, Checkbox, ConfirmDialog, IconButton, TextField, icons, useSnackbar, type MenuItem } from '@/ui';

/**
 * The parameter presets of a model: apply one, save the current parameters as one (for this model,
 * or the common ones for every model), make one the model's default (applied when it is picked),
 * delete one.
 */
const props = defineProps<{ model: string; params: Record<string, unknown>; options: Record<string, unknown> }>();
const emit = defineEmits<{ apply: [Preset] }>();
const { t } = useI18n();
const snackbar = useSnackbar();
const presets = usePresets(() => props.model);
const actions = usePresetActions();

const list = computed(() => presets.data.value ?? []);
const items = computed<MenuItem[]>(() => [
  ...list.value.map((p) => ({ id: `apply:${p.id}`, label: `${p.name}${p.is_default ? ` · ${t('presets.default')}` : ''}${p.model ? '' : ` · ${t('presets.any')}`}`, icon: icons.SlidersHorizontal })),
  { id: 'save', label: t('presets.save'), icon: icons.Save },
  ...(list.value.length ? [{ id: 'manage', label: t('presets.manage'), icon: icons.Settings }] : []),
]);

const saveOpen = ref(false);
const name = ref('');
const forModel = ref(true);
const makeDefault = ref(false);
const manageOpen = ref(false);
const deleting = ref<Preset | null>(null);
const commonOnly = computed(() => !Object.keys(props.options).length);
const deleteOpen = computed({ get: () => !!deleting.value, set: (v) => (deleting.value = v ? deleting.value : null) });

function onSelect(id: string) {
  if (id === 'save') {
    name.value = '';
    forModel.value = true;
    makeDefault.value = false;
    saveOpen.value = true;
  } else if (id === 'manage') manageOpen.value = true;
  else {
    const p = list.value.find((x) => `apply:${x.id}` === id);
    if (p) {
      emit('apply', p);
      snackbar.show(t('presets.applied', { name: p.name }));
    }
  }
}

async function save() {
  if (!name.value.trim()) return;
  const model = forModel.value || !commonOnly.value ? props.model : null;
  try {
    const p = await actions.create.mutateAsync({ name: name.value.trim(), model, params: props.params, engine_options: model ? props.options : {} });
    if (makeDefault.value && model) await actions.setDefault.mutateAsync({ id: p.id, value: true });
    snackbar.show(t('presets.saved', { name: p.name }));
    saveOpen.value = false;
  } catch (e) {
    snackbar.show(e instanceof Error ? e.message : String(e));
  }
}
const toggleDefault = (p: Preset) => actions.setDefault.mutate({ id: p.id, value: !p.is_default });
function confirmDelete() {
  if (deleting.value) actions.remove.mutate(deleting.value.id);
  deleting.value = null;
}
</script>

<template>
  <AppMenu :items="items" align="start" @select="onSelect">
    <template #default="{ toggle }">
      <AppButton variant="tonal" :icon="icons.SlidersHorizontal" @click="toggle">{{ t('presets.title') }}</AppButton>
    </template>
  </AppMenu>
  <AppDialog v-model:open="saveOpen" :title="t('presets.saveTitle')" width="small" :close-label="t('common.close')">
    <div class="form">
      <TextField v-model="name" :label="t('presets.name')" @enter="save" />
      <Checkbox v-if="commonOnly" v-model="forModel" :label="t('presets.forModel', { model })" />
      <p v-else class="type-body-small muted">{{ t('presets.withOptions', { model }) }}</p>
      <Checkbox v-if="forModel || !commonOnly" v-model="makeDefault" :label="t('presets.makeDefault')" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="saveOpen = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!name.trim()" :loading="actions.create.isPending.value" @click="save">{{ t('common.save') }}</AppButton>
    </template>
  </AppDialog>
  <AppDialog v-model:open="manageOpen" :title="t('presets.title')" width="small" :close-label="t('common.close')">
    <ul class="list">
      <li v-for="p in list" :key="p.id" class="row">
        <span class="text">
          <span class="type-body-large">{{ p.name }}</span>
          <span class="type-body-small muted">{{ p.model ?? t('presets.any') }}{{ p.is_default ? ` · ${t('presets.default')}` : '' }}</span>
        </span>
        <IconButton v-if="p.model" :icon="p.is_default ? icons.Star : icons.StarOff" :label="p.is_default ? t('presets.unsetDefault') : t('presets.setDefault')" @click="toggleDefault(p)" />
        <IconButton :icon="icons.Trash2" :label="t('common.delete')" @click="deleting = p" />
      </li>
    </ul>
    <template #actions>
      <AppButton variant="text" @click="manageOpen = false">{{ t('common.done') }}</AppButton>
    </template>
  </AppDialog>
  <ConfirmDialog
    v-model:open="deleteOpen"
    :title="t('common.delete')"
    :message="t('presets.deleteConfirm', { name: deleting?.name ?? '' })"
    :confirm-label="t('common.delete')"
    :cancel-label="t('common.cancel')"
    danger
    @confirm="confirmDelete"
  />
</template>

<style scoped>
.form { display: flex; flex-direction: column; gap: var(--app-space-3); }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
.list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: var(--app-space-1); }
.row { display: flex; align-items: center; gap: var(--app-space-2); }
.text { display: flex; flex-direction: column; flex: 1; min-width: 0; }
</style>
