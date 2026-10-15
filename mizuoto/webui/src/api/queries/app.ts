import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { SettingsView } from '@/api/types';

export const useMeta = () => useQuery({ queryKey: keys.meta, queryFn: () => unwrap(api.GET('/api/v1/app/meta')), staleTime: 60_000 });

export const useVersion = () => useQuery({ queryKey: keys.version, queryFn: () => unwrap(api.GET('/api/v1/app/version')), staleTime: Infinity });

export const useSettings = () => useQuery({ queryKey: keys.settings, queryFn: () => unwrap(api.GET('/api/v1/settings')) });

export function useUpdateSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patch: Record<string, unknown>) => unwrap(api.PATCH('/api/v1/settings', { body: patch })),
    onSuccess: (data: SettingsView) => {
      qc.setQueryData(keys.settings, data);
      qc.invalidateQueries({ queryKey: keys.meta });
    },
  });
}

export async function getClientState<T>(key: string): Promise<T | null> {
  return (await unwrap(api.GET('/api/v1/client-state/{key}', { params: { path: { key } } }))) as T | null;
}

export async function putClientState(key: string, value: unknown): Promise<void> {
  await unwrap(api.PUT('/api/v1/client-state/{key}', { params: { path: { key } }, body: value as never }));
}
