<script setup lang="ts">
import { computed, ref } from 'vue';
import { useRemoteActions, useRemoteHosts } from '@/api/queries/remote';
import type { S } from '@/api/types';
import ErrorNotice from '@/components/ErrorNotice.vue';
import { useI18n } from '@/i18n';
import { AppButton, Badge, ConfirmDialog, IconButton, Surface, Switch, TextField, icons, useSnackbar } from '@/ui';

/**
 * Other machines running ``mizuoto agent``: their state, devices and installed engines, and the form
 * to add one. An engine is sent to a host from its card (Models › Engines › Device).
 */
const { t } = useI18n();
const snackbar = useSnackbar();
const hosts = useRemoteHosts();
const actions = useRemoteActions();
const error = ref<unknown>(null);
const onError = (e: unknown) => (error.value = e);

const name = ref('');
const address = ref('');
const token = ref('');
const tls = ref(false);
const caFile = ref('');
const editing = ref<string | null>(null);
const canSave = computed(() => !!name.value.trim() && !!address.value.trim() && (!!token.value.trim() || !!editing.value));
function edit(h: S['RemoteHostStatus']) {
  editing.value = h.name;
  name.value = h.name;
  address.value = h.address;
  tls.value = h.tls;
  token.value = '';
}
function reset() {
  editing.value = null;
  name.value = address.value = token.value = caFile.value = '';
  tls.value = false;
}
function save() {
  error.value = null;
  actions.save.mutate(
    { name: name.value.trim(), body: { address: address.value.trim(), token: token.value.trim() || null, tls: tls.value, ca_file: caFile.value.trim() } },
    {
      onSuccess: (status) => {
        snackbar.show(status.state === 'online' ? t('remote.added', { name: status.name }) : t('remote.addedOffline', { name: status.name, error: status.error ?? status.state }));
        reset();
      },
      onError,
    },
  );
}
const removing = ref<string | null>(null);
function remove(n: string) {
  actions.remove.mutate(n, { onSuccess: (engines) => snackbar.show(engines.length ? t('remote.removedEngines', { name: n, engines: engines.join(', ') }) : t('remote.removed', { name: n })), onError });
  removing.value = null;
}
const tone = (state: string) => (state === 'online' ? 'primary' : state === 'refused' ? 'error' : 'warning');
</script>

<template>
  <div class="remote">
    <p class="type-body-medium muted">{{ t('remote.intro') }}</p>
    <div class="row-end">
      <AppButton variant="text" :icon="icons.RefreshCw" :loading="actions.check.isPending.value" @click="actions.check.mutate()">{{ t('remote.check') }}</AppButton>
    </div>
    <p v-if="hosts.isSuccess.value && !hosts.data.value?.length" class="type-body-medium muted">{{ t('remote.none') }}</p>
    <Surface v-for="h in hosts.data.value ?? []" :key="h.name" :level="1" shape="medium" class="host">
      <div class="head">
        <span class="type-title-medium grow name">{{ h.name }}</span>
        <Badge :value="t(`remote.states.${h.state}`)" :tone="tone(h.state)" />
        <IconButton :icon="icons.Pencil" :label="t('common.edit')" @click="edit(h)" />
        <IconButton :icon="icons.Trash2" :label="t('common.remove')" @click="removing = h.name" />
      </div>
      <p class="type-body-small muted">{{ h.address }}{{ h.tls ? ' · TLS' : '' }}{{ h.info ? ` · ${h.info.name} · Mizuoto ${h.info.version}` : '' }}</p>
      <p v-if="h.error" class="type-body-small warn">{{ h.error }}</p>
      <template v-if="h.info">
        <p class="type-body-small">{{ t('remote.devices') }}: {{ h.info.devices.map((d) => `${d.id}${d.name && d.id !== 'cpu' ? ' (' + d.name + ')' : ''}`).join(', ') }}</p>
        <p class="type-body-small">{{ t('remote.engines') }}: {{ h.info.engines.filter((e) => e.ready).map((e) => e.id).join(', ') || t('remote.noEngines') }}</p>
      </template>
      <p v-if="h.engines.length" class="type-body-small">{{ t('remote.usedBy') }}: {{ h.engines.join(', ') }}</p>
    </Surface>

    <h3 class="type-title-small sub">{{ editing ? t('remote.editHost', { name: editing }) : t('remote.add') }}</h3>
    <div class="grid">
      <TextField v-model="name" :label="t('remote.name')" :supporting-text="t('remote.nameHint')" :disabled="!!editing" />
      <TextField v-model="address" :label="t('remote.address')" placeholder="192.168.1.20:7870" />
      <TextField v-model="token" type="password" :label="t('remote.token')" :supporting-text="editing ? t('remote.tokenKeep') : t('remote.tokenHint')" autocomplete="off" />
    </div>
    <Switch v-model="tls" :label="t('remote.tls')" :supporting-text="t('remote.tlsHint')" />
    <TextField v-if="tls" v-model="caFile" :label="t('remote.caFile')" :supporting-text="t('remote.caFileHint')" />
    <ErrorNotice v-if="error" :error="error" />
    <div class="row-end">
      <AppButton v-if="editing" variant="text" @click="reset">{{ t('common.cancel') }}</AppButton>
      <AppButton :icon="icons.Server" :disabled="!canSave" :loading="actions.save.isPending.value" @click="save">{{ editing ? t('common.save') : t('remote.addButton') }}</AppButton>
    </div>
    <ConfirmDialog
      :open="removing !== null"
      :title="t('remote.remove')"
      :message="t('remote.removeConfirm', { name: removing ?? '' })"
      :confirm-label="t('common.remove')"
      :cancel-label="t('common.cancel')"
      danger
      @update:open="(o: boolean) => !o && (removing = null)"
      @confirm="removing && remove(removing)"
    />
  </div>
</template>

<style scoped>
.remote { display: flex; flex-direction: column; gap: var(--app-space-3); min-width: 0; }
.remote p { margin: 0; }
.host { display: flex; flex-direction: column; gap: var(--app-space-1); padding: var(--app-space-3) var(--app-space-4); min-width: 0; }
.head { display: flex; align-items: center; gap: var(--app-space-2); min-width: 0; }
.grow { flex: 1; min-width: 0; }
.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(220px, 100%), 1fr)); gap: var(--app-space-3); align-items: start; }
.row-end { display: flex; justify-content: flex-end; gap: var(--app-space-2); }
.warn { color: var(--md-sys-color-error); }
.sub { margin: var(--app-space-2) 0 0; }
</style>
