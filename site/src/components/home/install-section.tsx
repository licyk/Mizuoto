import Link from "fumadocs-core/link";
import { ArrowRight } from "lucide-react";

import HomeFooter from "@/components/home/home-footer";
import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

/** The release-build install from the README; the guide covers Docker and a source checkout. */
const INSTALL_COMMANDS = [
  "uv tool install mizuoto --find-links dist/",
  "mizuoto webui   # http://127.0.0.1:7869",
] as const;

export default function InstallSection() {
  const copy = useSiteCopy().home.install;

  return (
    <PageSection className="is-outro" id="install" labelledBy="install-title">
      <PageReveal>
        <div className="home-install-card page-interactive">
          <div className="home-install-copy">
            <span>INSTALL</span>
            <h2 id="install-title">{copy.title}</h2>
            <p>{copy.description}</p>
            <pre className="home-install-command">
              <code>{INSTALL_COMMANDS.map((command) => `$ ${command}`).join("\n")}</code>
            </pre>
          </div>
          <Link className="home-install-link page-interactive" href="/docs/installation">
            {copy.link}
            <ArrowRight className="page-interactive-part" aria-hidden="true" />
          </Link>
        </div>
      </PageReveal>
      <HomeFooter />
    </PageSection>
  );
}
