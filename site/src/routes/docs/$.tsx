import { createFileRoute, notFound } from "@tanstack/react-router";
import { createServerFn } from "@tanstack/react-start";
import { staticFunctionMiddleware } from "@tanstack/start-static-server-functions";
import { useFumadocsLoader } from "fumadocs-core/source/client";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/layouts/docs/page";
import { Suspense, use } from "react";

import { useMDXComponents } from "@/components/mdx";
import { useDocumentMetadata } from "@/lib/document-metadata";
import { DEFAULT_LOCALE, languageIndependentPath, LOCALES, type Locale } from "@/lib/i18n";
import { docsOptions } from "@/lib/layout.shared";
import { useSiteLocale } from "@/lib/site-copy";
import { docs, source } from "@/lib/source";

interface LocalizedPageData {
  description: string;
  headTitle: string;
  path: string;
  pageTree: Awaited<ReturnType<typeof source.serializePageTree>>;
}

function makePageTreeLanguageIndependent<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(makePageTreeLanguageIndependent) as T;
  }
  if (value === null || typeof value !== "object") return value;

  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [
      key,
      key === "url" && typeof entry === "string"
        ? languageIndependentPath(entry)
        : makePageTreeLanguageIndependent(entry),
    ]),
  ) as T;
}

export const Route = createFileRoute("/docs/$")({
  component: DocumentationPage,
  loader: async ({ params }) => {
    const slugs = params._splat?.split("/").filter(Boolean) ?? [];
    const data = await loadPage({ data: slugs });
    await Promise.all(LOCALES.map((locale) => docs.getPage(data[locale].path)?.preload()));
    return data;
  },
  head: ({ loaderData }) => {
    const data = loaderData?.[DEFAULT_LOCALE];
    return {
      meta: data
        ? [{ title: data.headTitle }, { name: "description", content: data.description }]
        : [],
    };
  },
});

const loadPage = createServerFn({ method: "GET" })
  .validator((slugs: string[]) => slugs)
  .middleware([staticFunctionMiddleware])
  .handler(async ({ data: slugs }) => {
    const entries = await Promise.all(
      LOCALES.map(async (locale): Promise<readonly [Locale, LocalizedPageData]> => {
        const page = source.getPage(slugs, locale);
        if (!page) throw notFound();

        return [
          locale,
          {
            description: page.data.description ?? "",
            headTitle:
              slugs.length === 0
                ? locale === "zh-CN"
                  ? "Mizuoto 文档"
                  : "Mizuoto Documentation"
                : `${page.data.title} — Mizuoto`,
            path: page.path,
            pageTree: makePageTreeLanguageIndependent(
              await source.serializePageTree(source.getPageTree(locale)),
            ),
          },
        ];
      }),
    );

    return Object.fromEntries(entries) as Record<Locale, LocalizedPageData>;
  });

function DocumentationContent({ path }: { path: string }) {
  const page = docs.getPage(path);
  if (!page) throw new Error(`Documentation page not found: ${path}`);

  const { toc } = use(page.load());
  const MDX = page.body;

  return (
    <DocsPage toc={toc}>
      <DocsTitle>{page.title}</DocsTitle>
      <DocsDescription>{page.description}</DocsDescription>
      <DocsBody>
        <MDX components={useMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

function DocumentationPage() {
  const locale = useSiteLocale();
  const localizedData = Route.useLoaderData()[locale];
  const { pageTree, path } = useFumadocsLoader(localizedData);

  useDocumentMetadata({
    description: localizedData.description,
    title: localizedData.headTitle,
  });

  return (
    <DocsLayout {...docsOptions(locale)} tree={pageTree}>
      <Suspense>
        <DocumentationContent key={path} path={path} />
      </Suspense>
    </DocsLayout>
  );
}
