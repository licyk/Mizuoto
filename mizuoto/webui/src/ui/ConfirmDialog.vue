<script setup lang="ts">
import AppButton from '@/ui/AppButton.vue';
import AppDialog from '@/ui/AppDialog.vue';

defineProps<{ title: string; message?: string; confirmLabel: string; cancelLabel: string; danger?: boolean; loading?: boolean }>();
const open = defineModel<boolean>('open', { default: false });
const emit = defineEmits<{ confirm: [] }>();
</script>

<template>
  <AppDialog v-model:open="open" :title="title" width="small">
    <p class="type-body-medium muted message">{{ message }}</p>
    <slot />
    <template #actions>
      <AppButton variant="text" @click="open = false">{{ cancelLabel }}</AppButton>
      <AppButton :variant="danger ? 'filled' : 'tonal'" :class="{ danger }" :loading="loading" @click="emit('confirm')">{{ confirmLabel }}</AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.message { margin: 0 0 var(--app-space-2); white-space: pre-line; }
.danger { --md-filled-button-container-color: var(--md-sys-color-error); --md-filled-button-label-text-color: var(--md-sys-color-on-error); }
</style>
