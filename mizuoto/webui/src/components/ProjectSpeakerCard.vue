<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useModelSpec } from '@/api/queries/models';
import type { SpeakerInput } from '@/api/types';
import ModelPicker from '@/components/ModelPicker.vue';
import ParamsPanel from '@/components/ParamsPanel.vue';
import VoicePicker from '@/components/VoicePicker.vue';
import { useI18n } from '@/i18n';
import { AppButton, AxisPanes, Badge, ExpansionPanel, IconButton, SegmentedButton, SelectField, Surface, TextField, icons } from '@/ui';

/**
 * One member of a project's cast: its name, the model it speaks with (the project's when it names
 * none), and its voice: a library voice, a preset speaker of the model, or the model's own.
 */
const props = defineProps<{ projectModel: string | null; narrator: boolean; lines: number }>();
const speaker = defineModel<SpeakerInput>({ required: true });
const emit = defineEmits<{ change: []; remove: [] }>();
const { t, tOr } = useI18n();
const model = computed(() => speaker.value.model || props.projectModel || null);
const spec = useModelSpec(model);
const caps = computed(() => new Set<string>(spec.data.value?.capabilities ?? []));

const KINDS = ['library', 'preset', 'auto'] as const;
type Kind = (typeof KINDS)[number];
const kinds = computed<Kind[]>(() => {
  const s = spec.data.value;
  if (!s) return ['library', 'preset', 'auto'];
  const out: Kind[] = [];
  if (s.voice_modes.some((m) => ['clone', 'clone_no_transcript', 'design', 'trained'].includes(m)) || s.speakers.length) out.push('library');
  if (s.speakers.length) out.push('preset');
  if (caps.value.has('auto_voice')) out.push('auto');
  return out.length ? out : ['library'];
});
const kind = ref<Kind>(speaker.value.voice ? 'library' : speaker.value.preset ? 'preset' : 'auto');
watch(
  () => [speaker.value.voice, speaker.value.preset],
  ([voice, preset]) => {
    if (voice) kind.value = 'library';
    else if (preset) kind.value = 'preset';
  },
);
const kindOptions = computed(() => kinds.value.map((k) => ({ value: k, label: t(`projects.voiceKinds.${k}`) })));

function update(patch: Partial<SpeakerInput>) {
  speaker.value = { ...speaker.value, ...patch };
  emit('change');
}
function setKind(k: Kind) {
  kind.value = k;
  if (k === 'auto') update({ voice: null, preset: null });
  else if (k === 'library') update({ preset: null });
  else update({ voice: null });
}
const voice = computed({ get: () => speaker.value.voice ?? null, set: (v) => update({ voice: v, preset: null }) });
const preset = computed({ get: () => speaker.value.preset ?? '', set: (v) => update({ preset: v || null, voice: null }) });
const speakerOptions = computed(() => (spec.data.value?.speakers ?? []).map((s) => ({ value: s, label: s })));
const ownModel = computed({ get: () => speaker.value.model ?? null, set: (v) => update({ model: v }) });
const language = computed({ get: () => speaker.value.language ?? '', set: (v) => update({ language: v || null }) });
const languageOptions = computed(() => [{ value: '', label: t('projects.projectLanguage') }, ...(spec.data.value?.languages ?? []).filter((l) => l !== 'auto').map((l) => ({ value: l, label: tOr(`languages.${l}`, l) }))]);
const params = computed({ get: () => (speaker.value.params ?? {}) as Record<string, unknown>, set: (v) => update({ params: v as SpeakerInput['params'] }) });
const options = computed({ get: () => (speaker.value.engine_options ?? {}) as Record<string, unknown>, set: (v) => update({ engine_options: v }) });
const paramsOpen = ref(false);
const changedParams = computed(() => Object.values(speaker.value.params ?? {}).filter((v) => v !== null && v !== undefined).length + Object.keys(speaker.value.engine_options ?? {}).length);
</script>

<template>
  <Surface :level="0" shape="large" class="speaker">
    <div class="head">
      <TextField :model-value="speaker.name" :label="t('projects.speakerName')" class="grow" @update:model-value="update({ name: String($event ?? '') })" />
      <Badge v-if="narrator" :value="t('projects.narrator')" tone="primary" />
      <span class="type-body-small muted">{{ t('projects.lines', { n: lines }) }}</span>
      <IconButton :icon="icons.Trash2" :label="t('common.remove')" @click="emit('remove')" />
    </div>
    <div class="line">
      <ModelPicker v-model="ownModel" :label="speaker.model ? t('projects.model') : t('projects.projectModelIs', { model: projectModel ?? '—' })" class="grow" />
      <AppButton v-if="speaker.model" variant="text" @click="ownModel = null">{{ t('projects.useProjectModel') }}</AppButton>
    </div>
    <SegmentedButton :model-value="kind" :options="kindOptions" @update:model-value="setKind($event as Kind)" />
    <!-- Each way to give a voice is a pane; a change slides the way the control moved. -->
    <AxisPanes :value="kind" :order="KINDS">
      <div v-if="kind === 'library'" key="library"><VoicePicker v-model="voice" :model="model" /></div>
      <div v-else-if="kind === 'preset'" key="preset"><SelectField v-model="preset" :label="t('speak.speaker')" :options="speakerOptions" /></div>
      <p v-else key="auto" class="type-body-medium muted auto">{{ t('speak.autoVoice') }}</p>
    </AxisPanes>
    <div class="grid">
      <SelectField v-model="language" :label="t('common.language')" :options="languageOptions" />
      <TextField v-if="caps.has('instruct')" :model-value="speaker.instruct ?? ''" :label="t('projects.instruct')" @update:model-value="update({ instruct: String($event ?? '') || null })" />
    </div>
    <ExpansionPanel v-model:open="paramsOpen" :label="t('projects.params')" :supporting-text="changedParams ? t('projects.paramsSet', { n: changedParams }) : ''" :icon="icons.SlidersHorizontal">
      <ParamsPanel v-model:params="params" v-model:options="options" :spec="spec.data.value ?? null" />
    </ExpansionPanel>
  </Surface>
</template>

<style scoped>
.speaker { display: flex; flex-direction: column; gap: var(--app-space-3); padding: var(--app-space-4); min-width: 0; }
.head, .line { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.grow { flex: 1; min-width: 0; }
.auto { margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: var(--app-space-3); }
</style>
