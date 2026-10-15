import { useQuery } from '@tanstack/vue-query';
import { computed, type MaybeRefOrGetter, toValue } from 'vue';
import { api, unwrap, uploadRaw } from '@/api/client';
import { keys } from '@/api/queries/keys';
import type { AudioFile, Peaks } from '@/api/types';

export const uploadAudio = (file: Blob & { name?: string }, onProgress?: (loaded: number, total: number) => void, signal?: AbortSignal) =>
  uploadRaw<AudioFile>('/api/v1/audio/files', { name: file.name || 'recording.wav' }, file, onProgress, signal);

/** Files on the server's disk (a browser on the server's own machine only). */
export const registerServerPaths = (paths: string[]) => unwrap(api.POST('/api/v1/audio/files/paths', { body: { paths } }));

export const transcribeAudio = (audioId: string, language?: string | null) => unwrap(api.POST('/api/v1/audio/transcribe', { body: { audio_id: audioId, language } }));

export const useBrowse = (path: MaybeRefOrGetter<string>, enabled: MaybeRefOrGetter<boolean> = true, dirsOnly: MaybeRefOrGetter<boolean> = false) =>
  useQuery({
    queryKey: computed(() => keys.browse(toValue(path), toValue(dirsOnly))),
    queryFn: () => unwrap(api.GET('/api/v1/audio/browse', { params: { query: { path: toValue(path) || undefined, dirs_only: toValue(dirsOnly) } } })),
    enabled: computed(() => toValue(enabled)),
  });

/** Where a waveform comes from: an audio file, an output, or a voice's reference clip (``voiceId/sha``). */
export type PeaksSource = { kind: 'file' | 'output' | 'reference'; id: string };

export function fetchPeaks(source: PeaksSource, points: number): Promise<Peaks> {
  const query = { points };
  if (source.kind === 'file') return unwrap(api.GET('/api/v1/audio/files/{audio_id}/peaks', { params: { path: { audio_id: source.id }, query } }));
  if (source.kind === 'output') return unwrap(api.GET('/api/v1/outputs/{output_id}/peaks', { params: { path: { output_id: source.id }, query } }));
  const [voiceId, sha] = source.id.split('/');
  return unwrap(api.GET('/api/v1/voices/{voice_id}/references/{sha}/peaks', { params: { path: { voice_id: voiceId, sha }, query } }));
}

/** Peaks are computed once on the server and never change for a file; cache them for the session. */
export const usePeaks = (source: MaybeRefOrGetter<PeaksSource | null>, points = 1024) =>
  useQuery({
    queryKey: computed(() => {
      const s = toValue(source);
      return keys.peaks(s?.kind ?? '', s?.id ?? '', points);
    }),
    queryFn: () => fetchPeaks(toValue(source)!, points),
    enabled: computed(() => !!toValue(source)),
    staleTime: Infinity,
  });
