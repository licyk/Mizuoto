<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { EmotionInput } from '@/api/types';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import type { InputItem } from '@/components/inputs';
import { useI18n } from '@/i18n';
import { ParamSlider, SegmentedControl, TextField } from '@/ui';

/**
 * How the speech should feel, in the ways the model takes it: a weight per emotion (sliders), a clip
 * whose emotion is borrowed (not its voice), with a strength, or the emotion described in words.
 * ``null`` when nothing is set.
 */
const props = defineProps<{ emotions: string[]; vector: boolean; audio: boolean; text: boolean }>();
const model = defineModel<EmotionInput | null>({ default: null });
const { t, tOr } = useI18n();

type Way = 'vector' | 'audio' | 'text';
const ways = computed<Way[]>(() => (['vector', 'audio', 'text'] as Way[]).filter((w) => props[w] && (w !== 'vector' || props.emotions.length)));
const way = ref<Way>(ways.value[0] ?? 'vector');
watch(ways, (list) => {
  if (list.length && !list.includes(way.value)) way.value = list[0]!;
});
const wayOptions = computed(() => ways.value.map((w) => ({ value: w, label: t(`speak.emotionWays.${w}`) })));

const weights = ref<Record<string, number>>({});
const clip = ref<InputItem[]>([]);
const strength = ref(1);
const described = ref('');

watch(
  [way, weights, clip, strength, described],
  () => {
    if (way.value === 'vector') {
      const set = Object.fromEntries(Object.entries(weights.value).filter(([, v]) => v > 0));
      model.value = Object.keys(set).length ? { vector: set } : null;
    } else if (way.value === 'audio') {
      const ready = clip.value.find((c) => c.status === 'ready' && c.audioId);
      model.value = ready ? { audio_id: ready.audioId, strength: strength.value } : null;
    } else model.value = described.value.trim() ? { text: described.value.trim() } : null;
  },
  { deep: true },
);

defineExpose({
  reset() {
    weights.value = {};
    clip.value = [];
    strength.value = 1;
    described.value = '';
  },
});
</script>

<template>
  <div v-if="ways.length" class="emotion">
    <SegmentedControl v-if="wayOptions.length > 1" v-model="way" :options="wayOptions" />
    <div v-if="way === 'vector'" class="grid">
      <ParamSlider
        v-for="e in emotions"
        :key="e"
        :model-value="weights[e] ?? 0"
        :label="tOr(`emotions.${e}`, e)"
        :min="0"
        :max="1"
        :step="0.05"
        :default-value="0"
        :reset-label="t('common.reset')"
        @update:model-value="weights = { ...weights, [e]: $event }"
      />
    </div>
    <template v-else-if="way === 'audio'">
      <AudioSourceInput v-model="clip" :multiple="false" :label="t('speak.emotionClip')" />
      <p class="type-body-small muted">{{ t('speak.emotionClipHint') }}</p>
      <ParamSlider v-model="strength" :label="t('speak.emotionStrength')" :min="0" :max="2" :step="0.05" :default-value="1" :reset-label="t('common.reset')" />
    </template>
    <TextField v-else v-model="described" :label="t('speak.emotionText')" :placeholder="t('speak.emotionTextPlaceholder')" />
  </div>
</template>

<style scoped>
.emotion { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
</style>
