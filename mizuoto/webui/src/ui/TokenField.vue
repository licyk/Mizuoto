<script setup lang="ts">
import { ref } from 'vue';
import AppButton from '@/ui/AppButton.vue';
import TextField from '@/ui/TextField.vue';
import { KeyRound } from '@/ui/icons';

/**
 * A write-only secret. The server never returns a token, only whether one is configured,
 * so this field shows that state and lets the user replace or clear it.
 */
defineProps<{ label: string; configured: boolean; saveLabel: string; clearLabel: string; configuredText: string; notConfiguredText: string }>();
const emit = defineEmits<{ save: [string]; clear: [] }>();
const value = ref('');

function save() {
  if (!value.value) return;
  emit('save', value.value);
  value.value = '';
}
</script>

<template>
  <div class="token-field">
    <TextField
      v-model="value"
      type="password"
      autocomplete="off"
      :label="label"
      :icon="KeyRound"
      :supporting-text="configured ? configuredText : notConfiguredText"
      @enter="save"
    />
    <div class="actions">
      <AppButton variant="tonal" :disabled="!value" @click="save">{{ saveLabel }}</AppButton>
      <AppButton v-if="configured" variant="text" @click="emit('clear')">{{ clearLabel }}</AppButton>
    </div>
  </div>
</template>

<style scoped>
.token-field { display: flex; flex-wrap: wrap; align-items: flex-start; gap: var(--app-space-2); }
.token-field > :first-child { flex: 1 1 280px; }
.actions { display: flex; gap: var(--app-space-2); padding-top: var(--app-space-2); }
</style>
