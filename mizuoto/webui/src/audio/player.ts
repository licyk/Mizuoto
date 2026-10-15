/**
 * One audio element for the whole page: playing a result stops whatever played before, and A/B
 * switches between a result and its source at the same position. Plain TypeScript over an
 * HTMLMediaElement, so it can be tested without a browser.
 */

export interface Track {
  /** Identifies the thing playing (an output id), so a list can show which row plays. */
  key: string;
  url: string;
  label: string;
  /** The other side of an A/B pair: usually the source the result was made from. */
  compareUrl?: string | null;
}

export type Side = 'a' | 'b';

export interface PlayerState {
  track: Track | null;
  side: Side;
  playing: boolean;
  time: number;
  duration: number;
}

type Listener = (state: PlayerState) => void;

export class AudioPlayer {
  readonly el: HTMLMediaElement;
  private state: PlayerState = { track: null, side: 'a', playing: false, time: 0, duration: 0 };
  private listeners = new Set<Listener>();
  private frame: number | null = null;

  constructor(el?: HTMLMediaElement) {
    this.el = el ?? new Audio();
    this.el.preload = 'metadata';
    // ``timeupdate`` fires only every 250 ms or so (up to a second in some browsers); while playing,
    // the position is read once a frame instead, so progress moves smoothly.
    this.el.addEventListener('timeupdate', () => this.patch({ time: this.el.currentTime }));
    this.el.addEventListener('durationchange', () => this.patch({ duration: Number.isFinite(this.el.duration) ? this.el.duration : 0 }));
    this.el.addEventListener('playing', () => this.follow());
    this.el.addEventListener('play', () => this.patch({ playing: true }));
    this.el.addEventListener('pause', () => this.unfollow({ playing: false }));
    this.el.addEventListener('ended', () => this.unfollow({ playing: false }));
    this.el.addEventListener('waiting', () => this.unfollow({}));
  }

  /** Read the position every frame while the element plays. */
  private follow() {
    if (this.frame !== null || typeof requestAnimationFrame === 'undefined') return;
    const tick = () => {
      this.frame = null;
      if (this.el.paused || this.el.ended) return;
      if (this.el.currentTime !== this.state.time) this.patch({ time: this.el.currentTime });
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  private unfollow(p: Partial<PlayerState>) {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.patch({ ...p, time: this.el.currentTime });
  }

  get snapshot(): PlayerState {
    return { ...this.state };
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => this.listeners.delete(listener);
  }

  private patch(p: Partial<PlayerState>) {
    this.state = { ...this.state, ...p };
    for (const l of this.listeners) l(this.snapshot);
  }

  private srcFor(track: Track, side: Side): string {
    return side === 'b' && track.compareUrl ? track.compareUrl : track.url;
  }

  /** Play ``track`` from the start (side A), or toggle it when it is the one loaded. */
  async play(track: Track, side: Side = 'a'): Promise<void> {
    if (this.state.track?.key === track.key && this.state.side === side) {
      await this.toggle();
      return;
    }
    this.patch({ track, side, time: 0, duration: 0 });
    this.el.src = this.srcFor(track, side);
    this.el.currentTime = 0;
    await this.safePlay();
  }

  async toggle(): Promise<void> {
    if (!this.state.track) return;
    if (this.el.paused) await this.safePlay();
    else this.el.pause();
  }

  /** Switch A/B at the same position, keeping play or pause. */
  async setSide(side: Side): Promise<void> {
    const track = this.state.track;
    if (!track || side === this.state.side || (side === 'b' && !track.compareUrl)) return;
    const time = this.el.currentTime;
    const wasPlaying = !this.el.paused;
    this.patch({ side });
    this.el.src = this.srcFor(track, side);
    const restore = () => {
      try {
        this.el.currentTime = time;
      } catch {
        /* not seekable yet */
      }
    };
    if (this.el.readyState >= 1) restore();
    else this.el.addEventListener('loadedmetadata', restore, { once: true });
    this.patch({ time });
    if (wasPlaying) await this.safePlay();
  }

  seek(seconds: number): void {
    this.el.currentTime = Math.max(0, seconds);
    this.patch({ time: this.el.currentTime });
  }

  stop(): void {
    this.el.pause();
    this.unfollow({});
    this.el.removeAttribute('src');
    this.patch({ track: null, playing: false, time: 0, duration: 0, side: 'a' });
  }

  private async safePlay(): Promise<void> {
    try {
      await this.el.play();
    } catch {
      // Autoplay refused or the source failed; the state stays paused.
      this.patch({ playing: false });
    }
  }
}
