<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import { tokenColor } from '@/ui/useTokenColor';

/**
 * A spectrogram on a canvas: ``data`` is ``rows × cols`` bytes (base64), row 0 the lowest band,
 * 0 quiet and 255 loud. Quiet is the surface, loud the primary colour, the loudest the tertiary,
 * read from the theme so it follows light and dark.
 */
const props = withDefaults(defineProps<{ rows: number; cols: number; data: string; label?: string; height?: number }>(), { label: '', height: 160 });
const canvas = ref<HTMLCanvasElement | null>(null);

function rgb(color: string): [number, number, number] {
  const probe = document.createElement('canvas').getContext('2d');
  if (!probe) return [0, 0, 0];
  probe.fillStyle = color;
  const hex = probe.fillStyle as string;
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (m) return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
  const parts = hex.match(/\d+(\.\d+)?/g)?.map(Number) ?? [0, 0, 0];
  return [parts[0], parts[1], parts[2]];
}

function draw() {
  const el = canvas.value;
  const ctx = el?.getContext('2d');
  if (!el || !ctx || !props.rows || !props.cols) return;
  const bytes = Uint8Array.from(atob(props.data), (c) => c.charCodeAt(0));
  el.width = props.cols;
  el.height = props.rows;
  const low = rgb(tokenColor(el, '--md-sys-color-surface-container-lowest', '#ffffff'));
  const mid = rgb(tokenColor(el, '--md-sys-color-primary', '#6750a4'));
  const high = rgb(tokenColor(el, '--md-sys-color-tertiary', '#7d5260'));
  const lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    const [a, b, f] = t < 0.75 ? [low, mid, t / 0.75] : [mid, high, (t - 0.75) / 0.25];
    for (let k = 0; k < 3; k++) lut[v * 3 + k] = a[k] + (b[k] - a[k]) * f;
  }
  const image = ctx.createImageData(props.cols, props.rows);
  for (let r = 0; r < props.rows; r++) {
    const y = props.rows - 1 - r; // row 0 (the lowest band) at the bottom
    for (let c = 0; c < props.cols; c++) {
      const v = bytes[r * props.cols + c] ?? 0;
      const i = (y * props.cols + c) * 4;
      image.data[i] = lut[v * 3];
      image.data[i + 1] = lut[v * 3 + 1];
      image.data[i + 2] = lut[v * 3 + 2];
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

onMounted(draw);
watch(() => [props.data, props.rows, props.cols], draw);
</script>

<template>
  <canvas ref="canvas" class="spectrogram" role="img" :aria-label="label" :style="{ height: `${height}px` }" />
</template>

<style scoped>
.spectrogram { display: block; width: 100%; image-rendering: auto; border-radius: var(--md-sys-shape-corner-small); background: var(--md-sys-color-surface-container-lowest); }
</style>
