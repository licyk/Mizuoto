<script setup lang="ts">
import { useQuery } from '@tanstack/vue-query';
import { computed, ref, watch } from 'vue';
import { api, unwrap } from '@/api/client';
import type { EngineStatus } from '@/api/types';
import { useI18n } from '@/i18n';
import { AppButton, AppDialog, Checkbox } from '@/ui';

/** A licence-gated engine's licence in full, and an explicit acceptance of that exact version. */
const props = defineProps<{ engine: EngineStatus | null }>();
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ accept: [string] }>();
const { t } = useI18n();
const agreed = ref(false);
watch(open, (o) => o && (agreed.value = false));
const licence = useQuery({
  queryKey: computed(() => ['engines', 'licence', props.engine?.id ?? '']),
  queryFn: () => unwrap(api.GET('/api/v1/engines/{engine_id}/licence', { params: { path: { engine_id: props.engine!.id } } })),
  enabled: computed(() => open.value && !!props.engine),
  staleTime: Infinity,
});
const key = () => (props.engine ? (props.engine.licence.version ? `${props.engine.licence.id}@${props.engine.licence.version}` : props.engine.licence.id) : '');
</script>

<template>
  <AppDialog v-model:open="open" :title="engine ? `${engine.title}: ${engine.licence.name}` : ''" width="medium" :close-label="t('common.close')">
    <div v-if="engine" class="body">
      <p class="type-body-medium">{{ t('engines.licenceText', { url: engine.licence.url || engine.licence.name }) }}</p>
      <a v-if="engine.licence.url" :href="engine.licence.url" target="_blank" rel="noopener" class="type-body-medium link">{{ engine.licence.url }}</a>
      <pre v-if="licence.data.value?.text" class="text type-body-small" tabindex="0" :aria-label="engine.licence.name">{{ licence.data.value.text }}</pre>
      <p v-if="engine.licence.notice" class="type-body-small muted">{{ engine.licence.notice }}</p>
      <Checkbox v-model="agreed" :label="t('engines.licenceAgree')" />
    </div>
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ t('common.cancel') }}</AppButton>
      <AppButton :disabled="!agreed" @click="(emit('accept', key()), (open = false))">{{ t('engines.acceptLicence') }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.body { display: flex; flex-direction: column; gap: var(--app-space-3); }
.link { color: var(--md-sys-color-primary); overflow-wrap: anywhere; }
.text { margin: 0; max-height: 40vh; overflow: auto; padding: var(--app-space-3); border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container); white-space: pre-wrap; font-family: inherit; }
p { margin: 0; }
</style>
