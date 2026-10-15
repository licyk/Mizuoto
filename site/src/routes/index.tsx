import { createFileRoute } from "@tanstack/react-router";
import { HomeLayout } from "fumadocs-ui/layouts/home";
import { useRef } from "react";

import FeaturesSection from "@/components/home/features-section";
import HeroSection from "@/components/home/hero-section";
import HighlightsSection from "@/components/home/highlights-section";
import InstallSection from "@/components/home/install-section";
import WorkflowSection from "@/components/home/workflow-section";
import SectionRail from "@/components/page/section-rail";
import { useDocumentMetadata } from "@/lib/document-metadata";
import { HOME_SECTION_IDS, homeSections } from "@/lib/home-sections";
import type { Locale } from "@/lib/i18n";
import { baseOptions } from "@/lib/layout.shared";
import { useActiveSection, useScrollReveal } from "@/lib/page-motion";
import { useSiteLocale } from "@/lib/site-copy";

const METADATA: Record<Locale, { description: string; title: string }> = {
  "zh-CN": {
    title: "Mizuoto — 开源语音合成引擎的统一平台",
    description:
      "CosyVoice、F5-TTS、GPT-SoVITS、OmniVoice、Qwen3-TTS、Fish Speech 与 IndexTTS，由同一个网页界面、命令行和 HTTP API 驱动：合成、克隆与设计音色，转换与编辑语音，长篇项目与模型微调。",
  },
  en: {
    title: "Mizuoto — one platform for open text-to-speech engines",
    description:
      "CosyVoice, F5-TTS, GPT-SoVITS, OmniVoice, Qwen3-TTS, Fish Speech and IndexTTS behind one web UI, one command line and one HTTP API: speak, clone and design voices, convert and edit speech, read long-form projects and fine-tune models.",
  },
};

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: METADATA["zh-CN"].title },
      { name: "description", content: METADATA["zh-CN"].description },
    ],
  }),
  component: Home,
});

function Home() {
  const locale = useSiteLocale();
  const homeRef = useRef<HTMLElement>(null);
  const sections = homeSections(locale);
  const activeSectionId = useActiveSection(HOME_SECTION_IDS);

  useDocumentMetadata(METADATA[locale]);
  useScrollReveal(homeRef);

  return (
    <HomeLayout {...baseOptions(locale)} className="home-page paged" ref={homeRef}>
      <SectionRail activeId={activeSectionId} sections={sections} />
      <HeroSection />
      <FeaturesSection />
      <WorkflowSection />
      <HighlightsSection />
      <InstallSection />
    </HomeLayout>
  );
}
