<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query';
import { computed, ref } from 'vue';
import { useI18n } from '@/i18n';
import { useAuthStore } from '@/stores/auth';
import { AppButton, AppDialog, TextField, icons } from '@/ui';

/** Asks for the access token when the server answers 401. */
const auth = useAuthStore();
const qc = useQueryClient();
const { t } = useI18n();
const value = ref('');
const open = computed({ get: () => auth.needed, set: (v) => (auth.needed = v) });

function submit() {
  if (!value.value) return;
  auth.setToken(value.value);
  value.value = '';
  qc.invalidateQueries();
}
</script>

<template>
  <AppDialog v-model:open="open" :title="t('auth.title')" width="small" :close-label="t('common.close')">
    <p class="type-body-medium muted">{{ t('auth.text') }}</p>
    <TextField v-model="value" type="password" :label="t('auth.token')" :icon="icons.KeyRound" autocomplete="current-password" @enter="submit" />
    <template #actions>
      <AppButton :disabled="!value" @click="submit">{{ t('common.save') }}</AppButton>
    </template>
  </AppDialog>
</template>
