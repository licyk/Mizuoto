import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

/** The models loaded in the server's process (kept fresh by the runtime_changed event). */
export const useRuntime = () => useQuery({ queryKey: keys.runtime, queryFn: () => unwrap(api.GET('/api/v1/runtime')), refetchInterval: 30_000 });

/** Unload a model, or (with none) every model not in use. */
export function useUnloadModels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (model: string | null) => unwrap(api.POST('/api/v1/runtime/unload', { body: { model } })),
    onSuccess: (status) => {
      qc.setQueryData(keys.runtime, status);
      qc.invalidateQueries({ queryKey: keys.computeUsage });
    },
  });
}
