import { mount } from '@vue/test-utils';
import { describe, expect, it } from 'vitest';
import { nextTick } from 'vue';
import IconButton from '@/ui/IconButton.vue';
import PickerMenu, { type PickerOption } from '@/ui/PickerMenu.vue';
import TextField from '@/ui/TextField.vue';

const options: PickerOption[] = [
  { value: 'default', label: 'System default', description: 'Follows the device the system uses' },
  { value: 'mic', label: 'Microphone (USB Audio Device)', description: 'Windows WASAPI · 1 ch · 48 kHz', badges: [{ text: 'default', tone: 'primary' }] },
  { value: 'cable', label: 'CABLE Output (VB-Audio Virtual Cable)', description: 'Windows WASAPI · 2 ch · 48 kHz', keywords: ['virtual'] },
  { value: 'gone', label: 'Old Headset', badges: [{ text: 'not connected', tone: 'warning' }], unavailable: true },
];

const picker = (props: Record<string, unknown> = {}) => mount(PickerMenu, { props: { label: 'Input', options, modelValue: 'mic', ...props }, attachTo: document.body });

describe('PickerMenu', () => {
  it('shows the choice with its second line and badges, and lists every option with its own', async () => {
    const w = picker();
    expect(w.find('.field .name').text()).toBe('Microphone (USB Audio Device)');
    expect(w.find('.field .meta').text()).toBe('Windows WASAPI · 1 ch · 48 kHz');
    expect(w.find('.field').text()).toContain('default');
    await w.find('.field').trigger('click');
    const rows = w.findAll('[role="option"]');
    expect(rows.map((r) => r.find('.name').text())).toEqual(options.map((o) => o.label));
    expect(rows[1].attributes('aria-selected')).toBe('true');
    expect(rows[0].find('.meta').text()).toBe('Follows the device the system uses');
    w.unmount();
  });

  it('picks an option and closes; a device that is gone cannot be picked', async () => {
    const w = picker();
    await w.find('.field').trigger('click');
    await w.findAll('[role="option"]')[3].trigger('click');
    expect(w.emitted('update:modelValue')).toBeUndefined();
    await w.findAll('[role="option"]')[2].trigger('click');
    expect(w.emitted('update:modelValue')).toEqual([['cable']]);
    expect(w.find('[role="listbox"]').exists()).toBe(false);
    w.unmount();
  });

  it('searches names, second lines and keywords once the list is long enough', async () => {
    const short = picker({ searchLabel: 'Search devices', searchFrom: 5 });
    await short.find('.field').trigger('click');
    expect(short.findComponent(TextField).exists()).toBe(false);
    short.unmount();

    const w = picker({ searchLabel: 'Search devices', searchFrom: 4, noMatches: 'No matches' });
    await w.find('.field').trigger('click');
    const search = (q: string) => w.findComponent(TextField).vm.$emit('update:modelValue', q);
    await search('virtual');
    expect(w.findAll('[role="option"]').map((r) => r.find('.name').text())).toEqual(['CABLE Output (VB-Audio Virtual Cable)']);
    await search('1 ch');
    expect(w.findAll('[role="option"]').map((r) => r.find('.name').text())).toEqual(['Microphone (USB Audio Device)']);
    await search('nothing like it');
    expect(w.findAll('[role="option"]')).toHaveLength(0);
    expect(w.find('.empty').text()).toBe('No matches');
    w.unmount();
  });

  it('while loading shows a spinner and the placeholder, and does not open', async () => {
    const w = picker({ options: [], modelValue: null, loading: true, placeholder: 'Looking for audio devices…' });
    expect(w.find('.field').attributes('aria-busy')).toBe('true');
    expect(w.find('.field .name').text()).toBe('Looking for audio devices…');
    expect(w.find('md-circular-progress').exists()).toBe(true);
    await w.find('.field').trigger('click');
    await nextTick();
    expect(w.find('[role="listbox"]').exists()).toBe(false);
    w.unmount();
  });
});

describe('IconButton', () => {
  it('becomes a link opening in a new tab with an href', () => {
    const link = mount(IconButton, { props: { icon: {}, label: 'Help', href: 'https://mizuoto.netlify.app/docs' } }).find('md-icon-button').element as HTMLElement & { href: string; target: string };
    expect([link.href, link.target]).toEqual(['https://mizuoto.netlify.app/docs', '_blank']);
  });
});
