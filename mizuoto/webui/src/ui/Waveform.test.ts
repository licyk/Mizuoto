import { mount } from '@vue/test-utils';
import { describe, expect, it, vi } from 'vitest';
import Waveform from '@/ui/Waveform.vue';

describe('Waveform', () => {
  it('seeks while the pointer drags, not only on release', async () => {
    // Without requestAnimationFrame each move seeks at once, so the test needs no timers.
    vi.stubGlobal('requestAnimationFrame', undefined);
    const wrapper = mount(Waveform, { props: { peaks: [-0.5, 0.5, -0.2, 0.2], progress: 0 } });
    const canvas = wrapper.find('canvas');
    canvas.element.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 48, right: 200, bottom: 48, x: 0, y: 0, toJSON: () => ({}) });
    await canvas.trigger('pointerdown', { button: 0, clientX: 20, pointerId: 1 });
    await canvas.trigger('pointermove', { clientX: 100, pointerId: 1 });
    expect(canvas.attributes('aria-valuenow')).toBe('50');
    await canvas.trigger('pointermove', { clientX: 300, pointerId: 1 });
    await canvas.trigger('pointerup', { clientX: 150, pointerId: 1 });
    expect(wrapper.emitted('seek')).toEqual([[0.1], [0.5], [1], [0.75]]);
    // After the release the cursor follows the player's progress again, and moves no longer seek.
    expect(canvas.attributes('aria-valuenow')).toBe('0');
    await canvas.trigger('pointermove', { clientX: 40, pointerId: 1 });
    expect(wrapper.emitted('seek')).toHaveLength(4);
    vi.unstubAllGlobals();
  });
});
