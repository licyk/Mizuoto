import { defineStore } from 'pinia';
import { onScopeDispose, shallowRef } from 'vue';
import { AudioPlayer, type PlayerState, type Side, type Track } from '@/audio/player';

/** The page's one player, as reactive state. */
export const usePlayerStore = defineStore('player', () => {
  const player = new AudioPlayer();
  const state = shallowRef<PlayerState>(player.snapshot);
  const unsubscribe = player.subscribe((s) => (state.value = s));
  onScopeDispose(unsubscribe);

  return {
    state,
    play: (track: Track, side?: Side) => player.play(track, side),
    toggle: () => player.toggle(),
    setSide: (side: Side) => player.setSide(side),
    seek: (seconds: number) => player.seek(seconds),
    stop: () => player.stop(),
    isPlaying: (key: string) => state.value.track?.key === key && state.value.playing,
  };
});
