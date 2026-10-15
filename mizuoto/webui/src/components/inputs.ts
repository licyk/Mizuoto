/** One clip given to Mizuoto: an upload, a server file or an earlier output, once it is an audio file. */
export interface InputItem {
  key: string;
  name: string;
  /** The audio file id once the clip is on the server (uploaded, registered or taken from an output). */
  audioId: string | null;
  duration?: number | null;
  status: 'uploading' | 'ready' | 'failed';
  progress?: number;
  error?: string;
  /** What is said in it, when known (typed, or filled by speech recognition). */
  transcript?: string | null;
}

let counter = 0;
export const inputKey = () => `in-${Date.now().toString(36)}-${counter++}`;

/** The audio file ids of the inputs that are ready. */
export const readyIds = (items: InputItem[]): string[] => items.filter((i) => i.status === 'ready' && i.audioId).map((i) => i.audioId as string);
