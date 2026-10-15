import { mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import { nextTick, ref } from 'vue';
import LevelMeter from '@/ui/LevelMeter.vue';
import ProgressBar from '@/ui/ProgressBar.vue';
import { useFalling } from '@/ui/motion/transitions';

describe('status bars', () => {
  it('draws a determinate progress bar itself, and leaves the indeterminate one to Material', async () => {
    const bar = mount(ProgressBar, { props: { value: 0.25, label: 'load' } });
    const fill = bar.find('.fill');
    expect(bar.find('[role="progressbar"]').attributes('aria-valuenow')).toBe('0.25');
    expect(fill.attributes('style')).toContain('--value: 0.25');
    await bar.setProps({ value: 3, tone: 'error' });
    expect(bar.find('.fill').attributes('style')).toContain('--value: 1');
    expect(bar.find('.track').classes()).toContain('error');
    await bar.setProps({ value: null });
    expect(bar.find('md-linear-progress').exists()).toBe(true);
  });

  it('moves the level meter by transform, marking a fall for the slower release', async () => {
    const meter = mount(LevelMeter, { props: { label: 'in', rmsDb: -30, peakDb: -12 } });
    expect(meter.find('.fill').attributes('style')).toContain('--level: 0.5');
    expect(meter.find('.fill').classes()).not.toContain('falling');
    await meter.setProps({ rmsDb: -45 });
    expect(meter.find('.fill').classes()).toContain('falling');
    await meter.setProps({ rmsDb: -20 });
    expect(meter.find('.fill').classes()).not.toContain('falling');
  });

  it('useFalling follows the direction of the last change', async () => {
    const v = ref(0.5);
    const falling = useFalling(() => v.value);
    v.value = 0.2;
    await nextTick();
    expect(falling.value).toBe(true);
    v.value = 0.9;
    await nextTick();
    expect(falling.value).toBe(false);
  });
});
