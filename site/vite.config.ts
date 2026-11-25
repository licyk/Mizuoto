import react from "@vitejs/plugin-react";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import tailwindcss from "@tailwindcss/vite";
import { fumadocsMdx } from "fumadocs-mdx/vite";
import { nitro } from "nitro/vite";
import { defineConfig } from "vite";

const base = process.env.VITE_BASE_PATH || "/";

export default defineConfig({
  base,
  server: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true,
  },
  plugins: [
    fumadocsMdx(),
    tailwindcss(),
    tanstackStart({
      spa: {
        enabled: true,
        maskPath: "/spa-shell",
        prerender: {
          enabled: true,
          crawlLinks: true,
        },
      },
      pages: [{ path: "/" }, { path: "/docs" }, { path: "/api/search" }],
    }),
    react(),
    nitro({
      preset: "node-server",
      // Vite's `base` rewrites the asset URLs but not where nitro serves the
      // public directory from; left apart, everything under a prefix 404s.
      baseURL: base,
    }),
  ],
  resolve: {
    tsconfigPaths: true,
  },
});
