import Link from "fumadocs-core/link";
import { Rss } from "lucide-react";

import { BilibiliIcon, GithubIcon } from "@/components/home/social-icons";
import { REPOSITORY_URL } from "@/lib/layout.shared";
import { useSiteCopy } from "@/lib/site-copy";

const AUTHOR = "licyk";

const SOCIAL_LINKS = [
  { key: "blog", href: "https://licyk.netlify.app", Icon: Rss },
  { key: "github", href: "https://github.com/licyk", Icon: GithubIcon },
  { key: "bilibili", href: "https://space.bilibili.com/46497516", Icon: BilibiliIcon },
] as const;

export default function HomeFooter() {
  const copy = useSiteCopy().home.footer;

  return (
    <footer className="home-footer">
      <div className="home-footer-layout">
        <div className="home-footer-identity">
          <strong>Mizuoto</strong>
          {/*
           * The page is prerendered, so the year is the build year until the
           * document is served again. Hydration must not warn about that.
           */}
          <p className="home-footer-copyright" suppressHydrationWarning>
            © {new Date().getFullYear()} {AUTHOR}
          </p>
        </div>
        <div className="home-footer-end">
          <nav className="home-footer-nav" aria-label={copy.label}>
            <Link className="home-footer-link page-interactive" href="/docs">
              {copy.docs}
            </Link>
            <Link className="home-footer-link page-interactive" href="/docs/quick-start">
              {copy.quickStart}
            </Link>
            <Link className="home-footer-link page-interactive" href="/docs/advanced-faq">
              {copy.faq}
            </Link>
            <Link className="home-footer-link page-interactive" href={REPOSITORY_URL}>
              {copy.repository}
            </Link>
          </nav>
          <nav className="home-footer-social" aria-label={copy.socialLabel}>
            {SOCIAL_LINKS.map(({ key, href, Icon }) => (
              <Link
                key={key}
                className="home-footer-social-link page-interactive"
                href={href}
                aria-label={copy.social[key]}
                title={copy.social[key]}
              >
                <Icon aria-hidden="true" />
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </footer>
  );
}
