<script setup lang="ts">
import { computed } from 'vue';
import type { StreamStatus } from '@/audio/streamPlayer';
import { formatDuration } from '@/format';
import { useI18n } from '@/i18n';
import { AppButton, AppIcon, ProgressBar, icons } from '@/ui';

/** The speech being streamed: what it is doing, how much has played and how much waits, and Stop. */
const props = defineProps<{ status: StreamStatus }>();
const emit = defineEmits<{ stop: [] }>();
const { t } = useI18n();
const live = computed(() => ['connecting', 'queued', 'playing', 'finishing'].includes(props.status.state));
const waiting = computed(() => props.status.state === 'connecting' || props.status.state === 'queued' || (props.status.state === 'playing' && !props.status.played));
const progress = computed(() => (props.status.received ? Math.min(1, props.status.played / props.status.received) : null));
</script>

<template>
  <div class="stream" role="status" aria-live="polite">
    <AppIcon :icon="live ? icons.AudioLines : icons.CircleCheck" :size="20" class="icon" :class="{ live }" />
    <div class="text">
      <span class="type-label-large">{{ t(`speak.streamStates.${status.state}`) }}</span>
      <span v-if="status.received" class="type-body-small muted">
        {{ t('speak.streamProgress', { played: formatDuration(status.played), ahead: formatDuration(status.buffered) }) }}
      </span>
      <ProgressBar v-if="live" :value="waiting ? null : progress" :label="t(`speak.streamStates.${status.state}`)" />
    </div>
    <AppButton v-if="live" variant="text" :icon="icons.Square" @click="emit('stop')">{{ t('speak.streamStop') }}</AppButton>
  </div>
</template>

<style scoped>
.stream { display: flex; align-items: center; gap: var(--app-space-3); padding: var(--app-space-2) var(--app-space-3); border-radius: var(--md-sys-shape-corner-medium); background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container); }
.icon { flex: none; }
.icon.live { color: var(--md-sys-color-primary); }
.text { display: flex; flex-direction: column; gap: var(--app-space-1); flex: 1; min-width: 0; }
.muted { color: inherit; opacity: 0.8; }
</style>
