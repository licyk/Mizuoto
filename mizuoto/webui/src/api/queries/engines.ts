import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

export const useEngines = () => useQuery({ queryKey: keys.engines, queryFn: () => unwrap(api.GET('/api/v1/engines')) });

export const useEngine = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.engine(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/engines/{engine_id}', { params: { path: { engine_id: toValue(id) ?? '' } } })),
    enabled: computed(() => !!toValue(id)),
  });

/** The doctor's report (torch against the hardware, the engines' packages): of one engine, or of all with none. */
export const useDoctor = (engine: MaybeRefOrGetter<string | null>, enabled: MaybeRefOrGetter<boolean> = true) =>
  useQuery({
    queryKey: computed(() => keys.doctor(toValue(engine))),
    queryFn: () => {
      const id = toValue(engine);
      return id ? unwrap(api.GET('/api/v1/engines/{engine_id}/doctor', { params: { path: { engine_id: id } } })) : unwrap(api.GET('/api/v1/runtime/doctor'));
    },
    enabled: computed(() => toValue(enabled)),
    staleTime: 60_000,
  });

export function useEngineActions() {
  const qc = useQueryClient();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: keys.engines });
    qc.invalidateQueries({ queryKey: keys.models });
  };
  const acceptLicence = useMutation({
    mutationFn: ({ engine, licence }: { engine: string; licence: string }) => unwrap(api.POST('/api/v1/engines/{engine_id}/licence', { params: { path: { engine_id: engine } }, body: { licence } })),
    onSuccess: refresh,
  });
  const configure = useMutation({
    mutationFn: ({ engine, patch }: { engine: string; patch: Record<string, unknown> }) => unwrap(api.PATCH('/api/v1/engines/{engine_id}', { params: { path: { engine_id: engine } }, body: patch })),
    onSuccess: refresh,
  });
  return { acceptLicence, configure };
}
