import { baseOptions } from "@/lib/layout.shared";
import { HomeLayout } from "fumadocs-ui/layouts/home";
import { DefaultNotFound } from "fumadocs-ui/layouts/home/not-found";
import { useSiteLocale } from "@/lib/site-copy";

export function NotFound() {
  const locale = useSiteLocale();

  return (
    <HomeLayout {...baseOptions(locale)}>
      <DefaultNotFound />
    </HomeLayout>
  );
}
