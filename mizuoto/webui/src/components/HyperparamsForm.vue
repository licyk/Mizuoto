<script setup lang="ts">
import { computed } from 'vue';
import type { OptionSpec } from '@/api/types';
import { useI18n } from '@/i18n';
import { SelectField, Switch, TextField } from '@/ui';

/**
 * A training method's hyperparameters: numbers as fields (their ranges span decades: 1e-7…1e-3,
 * 1…1 000 000), switches, menus, text. A value left empty is the method's default (``null``).
 */
const props = withDefaults(defineProps<{ specs: Record<string, OptionSpec>; group?: 'basic' | 'advanced' | 'all' }>(), { group: 'basic' });
const values = defineModel<Record<string, unknown>>({ default: () => ({}) });
const { t, tOr, tx } = useI18n();
const fields = computed(() => Object.entries(props.specs).filter(([, s]) => (props.group === 'all' ? s.group !== 'hidden' : s.group === props.group)));
const label = (name: string) => tOr(`train.hp.${name}`, tOr(`engines.options.${name}`, tx(name.replace(/_/g, ' '))));
const num = (v: number | null | undefined) => (v === null || v === undefined ? '' : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential() : String(v));
function help(spec: OptionSpec): string {
  const range = spec.min !== null && spec.max !== null && (spec.type === 'int' || spec.type === 'float') ? t('train.range', { min: num(spec.min), max: num(spec.max) }) : '';
  const fallback = spec.default !== null && spec.default !== undefined && spec.default !== '' ? t('train.default', { value: typeof spec.default === 'number' ? num(spec.default) : String(spec.default) }) : '';
  return [tx(spec.help), range, fallback].filter(Boolean).join(' · ');
}
const shown = (name: string, spec: OptionSpec) => values.value[name] ?? spec.default;
function set(name: string, spec: OptionSpec, raw: unknown) {
  let value: unknown = raw;
  if (spec.type === 'int' || spec.type === 'float') {
    const text = String(raw ?? '').trim();
    value = text === '' ? null : spec.type === 'int' ? Math.round(Number(text)) : Number(text);
    if (typeof value === 'number' && !Number.isFinite(value)) return;
  }
  const next = { ...values.value };
  if (value === null || value === '' || value === spec.default) delete next[name];
  else next[name] = value;
  values.value = next;
}
const invalid = (name: string, spec: OptionSpec) => {
  const v = values.value[name];
  if (typeof v !== 'number') return '';
  if ((spec.min !== null && v < spec.min) || (spec.max !== null && v > spec.max)) return t('train.outOfRange', { min: num(spec.min), max: num(spec.max) });
  return '';
};
</script>

<template>
  <div class="hp">
    <template v-for="[name, spec] in fields" :key="name">
      <Switch v-if="spec.type === 'bool'" :model-value="!!shown(name, spec)" :label="label(name)" :supporting-text="tx(spec.help)" @update:model-value="set(name, spec, $event)" />
      <SelectField
        v-else-if="spec.type === 'enum'"
        :model-value="String(shown(name, spec) ?? '')"
        :label="label(name)"
        :options="(spec.choices ?? []).map((c) => ({ value: c, label: c }))"
        :supporting-text="help(spec)"
        @update:model-value="set(name, spec, $event)"
      />
      <TextField
        v-else
        :model-value="name in values ? String(values[name]) : ''"
        :type="spec.type === 'str' ? 'text' : 'number'"
        :label="label(name)"
        :placeholder="spec.default === null || spec.default === undefined ? '' : String(spec.default)"
        :supporting-text="invalid(name, spec) ? '' : help(spec)"
        :error-text="invalid(name, spec)"
        @update:model-value="set(name, spec, $event)"
      />
    </template>
  </div>
</template>

<style scoped>
.hp { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(260px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-4); align-items: start; }
</style>
