import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';

export const useComputeUsage = () => useQuery({ queryKey: keys.computeUsage, queryFn: () => unwrap(api.GET('/api/v1/compute/usage')), refetchInterval: 30_000 });

export const useComputeInfo = () => useQuery({ queryKey: keys.computeInfo, queryFn: () => unwrap(api.GET('/api/v1/compute/info')), staleTime: 60_000 });

/** Free GPU memory: idle engines unload their models. */
export function useReleaseCompute() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST('/api/v1/compute/release')),
    onSuccess: (usage) => {
      qc.setQueryData(keys.computeUsage, usage);
      qc.invalidateQueries({ queryKey: keys.runtime });
    },
  });
}
