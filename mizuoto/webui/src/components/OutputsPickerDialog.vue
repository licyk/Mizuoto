<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useOutputs } from '@/api/queries/outputs';
import type { Output } from '@/api/types';
import { useI18n } from '@/i18n';
import { formatDuration } from '@/format';
import { AppButton, AppDialog, Checkbox, EmptyState, icons } from '@/ui';

/** Choose earlier results as inputs. */
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ select: [Output[]] }>();
const { t, tOr } = useI18n();
const outputs = useOutputs({}, open);
const chosen = ref<Set<string>>(new Set());
watch(open, (o) => o && (chosen.value = new Set()));
const items = computed(() => outputs.data.value?.items ?? []);

function toggle(id: string) {
  const next = new Set(chosen.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  chosen.value = next;
}
function add() {
  emit('select', items.value.filter((o) => chosen.value.has(o.id)));
  open.value = false;
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('outputsPicker.title')" width="medium" :close-label="t('common.close')">
    <EmptyState v-if="outputs.isSuccess.value && !items.length" :icon="icons.AudioLines" :title="t('outputsPicker.empty')" />
    <ul v-else class="list">
      <li v-for="o in items" :key="o.id" class="row">
        <Checkbox dense :model-value="chosen.has(o.id)" :label="o.name" @update:model-value="toggle(o.id)" />
        <span class="type-body-small muted meta">{{ o.model ?? tOr(`results.kinds.${o.kind}`, o.kind) }} · {{ o.duration ? formatDuration(o.duration) : '' }}</span>
      </li>
    </ul>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!chosen.size" @click="add">{{ t('outputsPicker.add', { n: chosen.size }) }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.list { list-style: none; margin: 0; padding: 0; max-height: 50vh; overflow: auto; }
.row { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.row > :first-child { flex: 1; min-width: 0; }
.meta { flex: none; }
</style>
