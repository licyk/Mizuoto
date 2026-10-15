<script setup lang="ts">
import '@material/web/fab/fab.js';
import type { Component } from 'vue';
import AppIcon from '@/ui/AppIcon.vue';

defineProps<{ icon: Component; label?: string; ariaLabel?: string }>();
defineEmits<{ click: [MouseEvent] }>();
</script>

<template>
  <md-fab :label="label ?? ''" :aria-label="ariaLabel ?? label" variant="primary" @click="$emit('click', $event)">
    <AppIcon slot="icon" :icon="icon" :size="24" />
  </md-fab>
</template>
