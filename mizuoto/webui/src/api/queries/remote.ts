import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';

/** The remote hosts (``mizuoto agent`` on other machines) and what each has. */
export const useRemoteHosts = () => useQuery({ queryKey: keys.remoteHosts, queryFn: () => unwrap(api.GET('/api/v1/remote/hosts')), staleTime: 15_000 });

export function useRemoteActions() {
  const qc = useQueryClient();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: keys.remoteHosts });
    qc.invalidateQueries({ queryKey: keys.engines });
    qc.invalidateQueries({ queryKey: keys.models });
    qc.invalidateQueries({ queryKey: keys.settings });
  };
  const check = useMutation({
    mutationFn: () => unwrap(api.GET('/api/v1/remote/hosts', { params: { query: { refresh: true } } })),
    onSuccess: (hosts) => qc.setQueryData(keys.remoteHosts, hosts),
  });
  const save = useMutation({
    mutationFn: ({ name, body }: { name: string; body: S['RemoteHostInput'] }) => unwrap(api.PUT('/api/v1/remote/hosts/{name}', { params: { path: { name } }, body })),
    onSuccess: refresh,
  });
  const remove = useMutation({ mutationFn: (name: string) => unwrap(api.DELETE('/api/v1/remote/hosts/{name}', { params: { path: { name } } })), onSuccess: refresh });
  const use = useMutation({
    mutationFn: ({ engine, host, device }: { engine: string; host: string | null; device: string }) =>
      unwrap(api.PUT('/api/v1/remote/engines/{engine_id}', { params: { path: { engine_id: engine } }, body: { host, device } })),
    onSuccess: refresh,
  });
  return { check, save, remove, use };
}
