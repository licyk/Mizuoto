import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { PresetInput } from '@/api/types';

/** The presets a model can use: its own and the common ones. */
export const usePresets = (model: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.presetList(toValue(model) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/presets', { params: { query: { model: toValue(model)! } } })),
    enabled: computed(() => !!toValue(model)),
  });

export function usePresetActions() {
  const qc = useQueryClient();
  const onSuccess = () => qc.invalidateQueries({ queryKey: keys.presets });
  return {
    create: useMutation({ mutationFn: (body: PresetInput) => unwrap(api.POST('/api/v1/presets', { body })), onSuccess }),
    remove: useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/presets/{preset_id}', { params: { path: { preset_id: id } } })), onSuccess }),
    setDefault: useMutation({
      mutationFn: ({ id, value }: { id: string; value: boolean }) => unwrap(api.POST('/api/v1/presets/{preset_id}/default', { params: { path: { preset_id: id } }, body: { default: value } })),
      onSuccess,
    }),
  };
}
