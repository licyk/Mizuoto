import { onBeforeUnmount, onMounted, ref } from 'vue';

/** Material 3 window size classes, by viewport width. */
export type WindowClass = 'compact' | 'medium' | 'expanded' | 'large' | 'extra-large';

export function windowClass(width: number): WindowClass {
  if (width < 600) return 'compact';
  if (width < 840) return 'medium';
  if (width < 1200) return 'expanded';
  if (width < 1600) return 'large';
  return 'extra-large';
}

export function useWindowClass() {
  const current = ref<WindowClass>(windowClass(typeof window === 'undefined' ? 1280 : window.innerWidth));
  const update = () => (current.value = windowClass(window.innerWidth));
  onMounted(() => window.addEventListener('resize', update, { passive: true }));
  onBeforeUnmount(() => window.removeEventListener('resize', update));
  return current;
}
