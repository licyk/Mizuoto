import { createRootRoute, HeadContent, Outlet, Scripts } from "@tanstack/react-router";

import favicon from "@/assets/favicon.svg?url&no-inline";
import { SiteProvider } from "@/components/site-provider";
import { DEFAULT_LOCALE } from "@/lib/i18n";
import appCss from "@/styles/app.css?url";

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      {
        title: "Mizuoto",
      },
      {
        name: "description",
        content:
          "Mizuoto：开源语音合成引擎的统一平台，一个网页界面、一套命令行和一个 HTTP API。",
      },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      {
        rel: "icon",
        type: "image/svg+xml",
        href: favicon,
      },
    ],
    scripts: [
      {
        src: "https://licyk-umami.netlify.app/script.js",
        async: true,
        "data-website-id": "9260f068-02e2-496e-b979-8f35577fbcbd",
      },
    ],
  }),
  component: RootComponent,
});

function RootComponent() {
  return (
    <html lang={DEFAULT_LOCALE} suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="flex min-h-screen flex-col">
        <SiteProvider>
          <Outlet />
        </SiteProvider>
        <Scripts />
      </body>
    </html>
  );
}
