import { onBeforeUnmount, ref, watch, type Ref } from 'vue';

/**
 * An element's height, kept current as it changes (a toolbar that wraps onto a second row on a
 * narrow screen). A height of 0, from an element out of the document, is ignored.
 */
export function useElementHeight(target: Ref<HTMLElement | null | undefined>) {
  const height = ref(0);
  let observer: ResizeObserver | null = null;
  const measure = (el: HTMLElement) => {
    if (el.offsetHeight) height.value = el.offsetHeight;
  };
  watch(
    target,
    (el) => {
      observer?.disconnect();
      observer = null;
      if (!el) return;
      measure(el);
      if (typeof ResizeObserver === 'undefined') return;
      observer = new ResizeObserver(() => measure(el));
      observer.observe(el);
    },
    { immediate: true, flush: 'post' },
  );
  onBeforeUnmount(() => observer?.disconnect());
  return height;
}
