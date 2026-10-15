import { useMutation, useQuery, useQueryClient } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap } from '@/api/client';
import { BASE_URL } from '@/api/baseUrl';
import { keys } from '@/api/queries/keys';
import type { S } from '@/api/types';
import { useJobsStore } from '@/stores/jobs';

export const useProjects = () => useQuery({ queryKey: keys.projects, queryFn: () => unwrap(api.GET('/api/v1/projects')) });

export const useProject = (id: MaybeRefOrGetter<string | null>) =>
  useQuery({
    queryKey: computed(() => keys.project(toValue(id) ?? '')),
    queryFn: () => unwrap(api.GET('/api/v1/projects/{project_id}', { params: { path: { project_id: toValue(id) ?? '' } } })),
    enabled: computed(() => !!toValue(id)),
  });

export const projectUrls = {
  subtitles: (outputId: string, format: 'srt' | 'vtt' = 'srt') => `${BASE_URL}/api/v1/projects/exports/${encodeURIComponent(outputId)}/subtitles?format=${format}`,
};

/** The synthesis request of a line, to try it in Speak. */
export const fetchLineRequest = (project: string, line: string) =>
  unwrap(api.GET('/api/v1/projects/{project_id}/lines/{line_id}/request', { params: { path: { project_id: project, line_id: line } } }));

export function useProjectActions() {
  const qc = useQueryClient();
  const jobs = useJobsStore();
  const path = (id: string) => ({ params: { path: { project_id: id } } });
  const refresh = (p?: S['Project']) => {
    qc.invalidateQueries({ queryKey: keys.projects });
    if (p) qc.setQueryData(keys.project(p.id), p);
  };
  const create = useMutation({ mutationFn: (body: S['ProjectInput']) => unwrap(api.POST('/api/v1/projects', { body })), onSuccess: refresh });
  const save = useMutation({ mutationFn: ({ id, body }: { id: string; body: S['ProjectInput'] }) => unwrap(api.PUT('/api/v1/projects/{project_id}', { ...path(id), body })), onSuccess: refresh });
  const remove = useMutation({ mutationFn: (id: string) => unwrap(api.DELETE('/api/v1/projects/{project_id}', path(id))), onSuccess: () => refresh() });
  const importScript = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['ScriptImport'] }) => unwrap(api.POST('/api/v1/projects/{project_id}/script', { ...path(id), body })),
    onSuccess: refresh,
  });
  const render = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['RenderRequest'] }) => unwrap(api.POST('/api/v1/projects/{project_id}/render', { ...path(id), body })),
    onSuccess: (job) => jobs.upsert(job, 'projects'),
  });
  const exportAudio = useMutation({
    mutationFn: ({ id, body }: { id: string; body: S['ExportRequest'] }) => unwrap(api.POST('/api/v1/projects/{project_id}/export', { ...path(id), body })),
    onSuccess: (job) => jobs.upsert(job, 'projects'),
  });
  const useTake = useMutation({
    mutationFn: ({ id, line, output }: { id: string; line: string; output: string }) =>
      unwrap(api.PUT('/api/v1/projects/{project_id}/lines/{line_id}/take', { params: { path: { project_id: id, line_id: line } }, body: { output_id: output } })),
    onSuccess: refresh,
  });
  return { create, save, remove, importScript, render, exportAudio, useTake };
}
