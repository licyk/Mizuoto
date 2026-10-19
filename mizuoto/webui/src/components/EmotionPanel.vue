<script setup lang="ts">
import { computed } from 'vue';
import type { ModelParamSpecs } from '@/api/types';
import AudioSourceInput from '@/components/AudioSourceInput.vue';
import type { InputItem } from '@/components/inputs';
import { EMOTION_WAYS, emotionWays, emptyEmotion, shownWay, type EmotionForm, type EmotionWay } from '@/components/speakForm';
import { useI18n } from '@/i18n';
import { AxisPanes, ParamSlider, SegmentedControl, TextField } from '@/ui';

/**
 * How the speech should feel, in the ways the model takes it: a weight per emotion (sliders), a clip
 * whose emotion is borrowed (not its voice), with a strength, or the emotion described in words. The
 * values are the parent's (``speakForm``), which sends only the way shown here.
 */
const props = defineProps<{ spec: ModelParamSpecs }>();
const form = defineModel<EmotionForm>({ default: emptyEmotion });
const { t, tOr } = useI18n();

const ways = computed(() => emotionWays(props.spec));
const way = computed({ get: () => shownWay(form.value, ways.value) ?? 'vector', set: (w: EmotionWay) => update({ way: w }) });
const wayOptions = computed(() => ways.value.map((w) => ({ value: w, label: t(`speak.emotionWays.${w}`) })));
const update = (patch: Partial<EmotionForm>) => (form.value = { ...form.value, ...patch });
const clip = computed({ get: () => form.value.clip, set: (c: InputItem[]) => update({ clip: c }) });
const strength = computed({ get: () => form.value.strength, set: (v: number) => update({ strength: v }) });
const described = computed({ get: () => form.value.described, set: (v: string) => update({ described: v }) });
</script>

<template>
  <div v-if="ways.length" class="emotion">
    <SegmentedControl v-if="wayOptions.length > 1" v-model="way" :options="wayOptions" />
    <!-- Each way is a pane; a change slides the way the control moved. -->
    <AxisPanes :value="way" :order="EMOTION_WAYS">
      <div v-if="way === 'vector'" key="vector" class="grid">
        <ParamSlider
          v-for="e in spec.emotions"
          :key="e"
          :model-value="form.weights[e] ?? 0"
          :label="tOr(`emotions.${e}`, e)"
          :min="0"
          :max="1"
          :step="0.05"
          :default-value="0"
          :reset-label="t('common.reset')"
          @update:model-value="update({ weights: { ...form.weights, [e]: $event } })"
        />
      </div>
      <div v-else-if="way === 'audio'" key="audio" class="pane">
        <AudioSourceInput v-model="clip" :multiple="false" :label="t('speak.emotionClip')" />
        <p class="type-body-small muted">{{ t('speak.emotionClipHint') }}</p>
        <ParamSlider v-model="strength" :label="t('speak.emotionStrength')" :min="0" :max="2" :step="0.05" :default-value="1" :reset-label="t('common.reset')" />
      </div>
      <div v-else key="text" class="pane">
        <TextField v-model="described" :label="t('speak.emotionText')" :placeholder="t('speak.emotionTextPlaceholder')" />
      </div>
    </AxisPanes>
  </div>
</template>

<style scoped>
.emotion { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.pane { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(280px, 100%), 1fr)); gap: var(--app-space-3) var(--app-space-6); }
.muted { margin: 0; color: var(--md-sys-color-on-surface-variant); }
</style>
