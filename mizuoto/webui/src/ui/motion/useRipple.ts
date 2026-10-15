import { onBeforeUnmount, onMounted, type Ref } from 'vue';
import { prefersReducedMotion } from '@/ui/motion/transitions';

/** A ripple from the pointer position on press, for project-built interactive components. */
export function useRipple(target: Ref<HTMLElement | null | undefined>) {
  const onPointerDown = (event: PointerEvent) => {
    const el = target.value;
    if (!el || event.button !== 0 || prefersReducedMotion()) return;
    const rect = el.getBoundingClientRect();
    const size = Math.hypot(rect.width, rect.height) * 2;
    const wave = document.createElement('span');
    wave.className = 'ripple-wave';
    wave.style.width = wave.style.height = `${size}px`;
    wave.style.left = `${event.clientX - rect.left - size / 2}px`;
    wave.style.top = `${event.clientY - rect.top - size / 2}px`;
    el.appendChild(wave);
    wave.addEventListener('animationend', () => wave.remove(), { once: true });
  };
  onMounted(() => target.value?.addEventListener('pointerdown', onPointerDown));
  onBeforeUnmount(() => target.value?.removeEventListener('pointerdown', onPointerDown));
}
