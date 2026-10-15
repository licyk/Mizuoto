import { access, copyFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const publicDirectory = fileURLToPath(new URL("../.output/public/", import.meta.url));
const metadata = JSON.parse(
  await readFile(fileURLToPath(new URL("../content/docs/meta.json", import.meta.url)), "utf8"),
);
const documentSlugs = metadata.pages.filter((page) => !page.startsWith("---"));
const documents = documentSlugs.map((slug) =>
  slug === "index" ? "docs/index.html" : `docs/${slug}/index.html`,
);
const requiredFiles = [
  "_shell.html",
  "index.html",
  "api/search",
  ...documents,
];

await Promise.all(requiredFiles.map((file) => access(join(publicDirectory, file))));

await copyFile(join(publicDirectory, "_shell.html"), join(publicDirectory, "404.html"));
