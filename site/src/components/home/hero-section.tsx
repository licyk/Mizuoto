import Link from "fumadocs-core/link";
import { ArrowRight, ChevronDown, Sparkles } from "lucide-react";

import PageSection from "@/components/page/page-section";
import ProductWorkspacePreview from "@/components/product-workspace-preview";
import { scrollToSection } from "@/lib/page-motion";
import { nextHomeSection } from "@/lib/home-sections";
import { useSiteCopy, useSiteLocale } from "@/lib/site-copy";

export default function HeroSection() {
  const locale = useSiteLocale();
  const copy = useSiteCopy().home.hero;
  // Resolved from the manifest so the button always advances exactly one
  // section, whatever follows the hero.
  const next = nextHomeSection("hero", locale);

  return (
    <PageSection
      backdropClassName="home-hero-backdrop"
      className="is-hero"
      id="hero"
      labelledBy="home-title"
    >
      <div className="home-hero-layout">
        <div className="home-hero-copy">
          <div className="home-eyebrow">
            <Sparkles aria-hidden="true" />
            {copy.eyebrow}
          </div>
          <h1 id="home-title">
            {copy.title}
            <span>{copy.titleAccent}</span>
          </h1>
          <p>{copy.description}</p>
          <div className="home-hero-actions">
            <Link className="home-primary-link page-interactive" href="/docs/quick-start">
              {copy.start}
              <ArrowRight className="page-interactive-part" aria-hidden="true" />
            </Link>
            {next ? (
              <button
                className="home-secondary-link page-interactive"
                onClick={() => scrollToSection(next.id)}
                type="button"
              >
                {copy.learn}
                {next.railLabel}
                <ChevronDown className="page-interactive-part" aria-hidden="true" />
              </button>
            ) : null}
          </div>
          <div className="home-platforms" aria-label={copy.platformsLabel}>
            <span>Windows</span>
            <span>macOS</span>
            <span>Linux</span>
          </div>
        </div>
        <ProductWorkspacePreview />
      </div>
    </PageSection>
  );
}
