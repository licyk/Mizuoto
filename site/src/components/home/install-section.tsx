import Link from "fumadocs-core/link";
import { ArrowRight } from "lucide-react";

import HomeFooter from "@/components/home/home-footer";
import PageReveal from "@/components/page/page-reveal";
import PageSection from "@/components/page/page-section";
import { useSiteCopy } from "@/lib/site-copy";

/** Points to the installation guide, which holds the commands: the right torch build depends on the machine. */
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
          </div>
          <Link className="home-install-link page-interactive" href={copy.href}>
            {copy.link}
            <ArrowRight className="page-interactive-part" aria-hidden="true" />
          </Link>
        </div>
      </PageReveal>
      <HomeFooter />
    </PageSection>
  );
}
