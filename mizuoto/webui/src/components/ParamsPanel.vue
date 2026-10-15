<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ModelParamSpecs, ParamField } from '@/api/types';
import ParamInput from '@/components/ParamInput.vue';
import { useI18n } from '@/i18n';
import { ExpansionPanel, icons } from '@/ui';

/**
 * A model's parameters, rendered from its spec: the common ones it takes, then its own options
 * under "Engine options". A value left alone is not sent, so the engine uses its own default. The
 * seed has its own field (Speak).
 */
const props = defineProps<{ spec: ModelParamSpecs | null }>();
const params = defineModel<Record<string, unknown>>('params', { default: () => ({}) });
const options = defineModel<Record<string, unknown>>('options', { default: () => ({}) });
const { t } = useI18n();
const advancedOpen = ref(false);

const fields = computed(() => (props.spec?.params ?? []).filter((f) => f.name !== 'seed' && f.group !== 'hidden'));
const main = computed(() => fields.value.filter((f) => f.kind === 'common' || f.group === 'basic'));
const advanced = computed(() => fields.value.filter((f) => f.kind === 'option' && f.group === 'advanced'));

const store = (f: ParamField) => (f.kind === 'common' ? params : options);
const get = (f: ParamField) => store(f).value[f.name] ?? null;
function set(f: ParamField, v: unknown) {
  const next = { ...store(f).value };
  if (v === null || v === undefined) delete next[f.name];
  else next[f.name] = v;
  store(f).value = next;
}
</script>

<template>
  <div class="params">
    <div v-if="main.length" class="grid">
      <ParamInput v-for="f in main" :key="`${f.kind}:${f.name}`" :field="f" :model-value="get(f)" @update:model-value="set(f, $event)" />
    </div>
    <ExpansionPanel v-if="advanced.length" v-model:open="advancedOpen" :label="t('params.advanced')" :icon="icons.SlidersHorizontal">
      <div class="grid">
        <ParamInput v-for="f in advanced" :key="`${f.kind}:${f.name}`" :field="f" :model-value="get(f)" @update:model-value="set(f, $event)" />
      </div>
    </ExpansionPanel>
  </div>
</template>

<style scoped>
.params { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); align-items: start; }
</style>
