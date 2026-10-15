<script setup lang="ts">
import { computed } from 'vue';
import type { Track } from '@/audio/player';
import { useI18n } from '@/i18n';
import { usePlayerStore } from '@/stores/player';
import { SegmentedControl } from '@/ui';

/** A/B between a result and its source at the same position, on the page's one player. */
const props = defineProps<{ track: Track }>();
const player = usePlayerStore();
const { t } = useI18n();
const loaded = computed(() => player.state.track?.key === props.track.key);
const side = computed({
  get: () => (loaded.value ? player.state.side : 'a'),
  set: (s: 'a' | 'b') => (loaded.value ? player.setSide(s) : player.play(props.track, s)),
});
const options = computed(() => [
  { value: 'a' as const, label: t('results.a') },
  { value: 'b' as const, label: t('results.b') },
]);
</script>

<template>
  <SegmentedControl v-model="side" :options="options" :aria-label="t('results.abLabel')" />
</template>
