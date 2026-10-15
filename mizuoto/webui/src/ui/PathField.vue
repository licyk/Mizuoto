<script setup lang="ts">
import TextField from '@/ui/TextField.vue';
import { Folder } from '@/ui/icons';

/** An absolute folder or file path on the server's machine. */
defineProps<{ label: string; supportingText?: string; errorText?: string; placeholder?: string }>();
const model = defineModel<string>({ default: '' });
defineEmits<{ enter: [] }>();
</script>

<template>
  <TextField
    v-model="model"
    :label="label"
    :icon="Folder"
    :placeholder="placeholder ?? '/path/to/models'"
    :supporting-text="supportingText"
    :error-text="errorText"
    autocomplete="off"
    @enter="$emit('enter')"
  />
</template>
