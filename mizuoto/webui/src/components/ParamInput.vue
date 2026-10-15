<script setup lang="ts">
import { computed } from 'vue';
import type { ParamField } from '@/api/types';
import { useI18n } from '@/i18n';
import { ParamSlider, SelectField, Switch, TextField } from '@/ui';

/**
 * One parameter of a model, by its type: a slider for numbers, a switch for a boolean with a
 * default, a three-way choice (engine default, yes, no) for one without, a menu for an enum. The
 * value ``null`` means "not set": the engine's own default applies.
 */
const props = defineProps<{ field: ParamField }>();
const model = defineModel<unknown>();
const { t, tOr, tx } = useI18n();
const f = computed(() => props.field);

const label = computed(() => (f.value.kind === 'common' ? tOr(`params.names.${f.value.name}`, f.value.name) : tx(f.value.name.replace(/_/g, ' '))));
const help = computed(() => [tx(f.value.help), f.value.approximate ? t('params.approximate') : ''].filter(Boolean).join(' '));
const hasDefault = computed(() => f.value.default !== null && f.value.default !== undefined);
const shown = computed(() => (model.value === null || model.value === undefined ? f.value.default : model.value));
const numeric = computed(() => Number(shown.value ?? f.value.min ?? 0));
const step = computed(() => f.value.step ?? (f.value.type === 'int' ? 1 : 0.01));

function set(v: unknown) {
  model.value = v === '' || v === undefined || (hasDefault.value && v === f.value.default) ? null : v;
}
const enumChoices = computed(() => [...(hasDefault.value ? [] : [{ value: '', label: t('params.engineDefault') }]), ...(f.value.choices ?? []).map((c) => ({ value: c, label: c }))]);
const boolChoices = computed(() => [
  { value: '', label: t('params.engineDefault') },
  { value: 'true', label: t('common.yes') },
  { value: 'false', label: t('common.no') },
]);
</script>

<template>
  <ParamSlider
    v-if="f.type === 'float' || f.type === 'int'"
    :model-value="numeric"
    :label="label"
    :help="help"
    :min="f.min ?? 0"
    :max="f.max ?? 1"
    :step="step"
    :default-value="typeof f.default === 'number' ? f.default : null"
    :reset-label="t('common.reset')"
    @update:model-value="set(f.type === 'int' ? Math.round($event) : $event)"
  />
  <Switch v-else-if="f.type === 'bool' && hasDefault" :model-value="!!shown" :label="label" :supporting-text="help" @update:model-value="set($event)" />
  <SelectField v-else-if="f.type === 'bool'" :model-value="shown === null || shown === undefined ? '' : String(shown)" :label="label" :options="boolChoices" :supporting-text="help" @update:model-value="set($event === '' ? null : $event === 'true')" />
  <SelectField v-else-if="f.type === 'enum'" :model-value="String(shown ?? '')" :label="label" :options="enumChoices" :supporting-text="help" @update:model-value="set($event || null)" />
  <TextField v-else :model-value="String(shown ?? '')" :label="label" :supporting-text="help" @update:model-value="set($event)" />
</template>
