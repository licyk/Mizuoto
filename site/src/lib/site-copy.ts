import { useI18n } from "fumadocs-ui/contexts/i18n";

import { DEFAULT_LOCALE, isLocale, type Locale } from "@/lib/i18n";

const zhCN = {
  pageRailLabel: "页面区块导航",
  theme: {
    hint: "点击切换主题",
    modes: { system: "主题：跟随系统", light: "主题：浅色", dark: "主题：深色" },
  },
  home: {
    sections: ["首页", "功能", "使用流程", "特点", "安装"],
    hero: {
      eyebrow: "开源语音合成引擎的统一平台",
      title: "七个语音合成引擎，",
      titleAccent: "同一个平台",
      description:
        "Mizuoto 把 CosyVoice、F5-TTS、GPT-SoVITS、OmniVoice、Qwen3-TTS、Fish Speech 和 IndexTTS 收进一个网页界面、一套命令行和一个 HTTP 接口：合成语音、克隆与设计音色、转换与编辑语音、朗读长篇作品，并用你自己的录音微调模型。",
      start: "开始使用",
      learn: "了解",
      platformsLabel: "支持的平台",
    },
    features: {
      title: "从一句话到一整本书",
      description: "每个界面都有对应的教程；同样的操作也都可以在命令行和 HTTP 接口中完成。",
      open: "查看教程",
      items: [
        {
          title: "合成",
          description:
            "任选一个已安装的模型，用音色库中的音色、一段参考音频、预置说话人或一句描述来说话；支持风格指令与情感，可边生成边播放。",
          href: "/docs/guide-speak",
        },
        {
          title: "项目",
          description: "长篇作品按章节组织，一组角色各用自己的模型与音色，逐句合成，导出整段音频与字幕。",
          href: "/docs/guide-projects",
        },
        {
          title: "工具",
          description: "音色转换（CosyVoice）保留原话的内容与节奏换一个声音；语音编辑（F5-TTS）用同一音色改写其中几句。",
          href: "/docs/guide-tools",
        },
        {
          title: "音色",
          description: "参考音频（缺少的文本由语音识别补全）、设计出的音色、预置说话人与专属微调权重，都在一个音色库里。",
          href: "/docs/guide-voices",
        },
        {
          title: "训练",
          description: "从录音建立数据集，按各引擎自己的流程逐阶段微调，实时查看损失与试听，得到音色专属权重或新模型。",
          href: "/docs/guide-train",
        },
      ],
    },
    workflow: {
      title: "从安装到第一段语音",
      description: "四步走完整个流程，每一步在网页界面和命令行中都有对应做法。",
      steps: [
        {
          title: "安装 Mizuoto",
          description: "从 PyPI、Docker 镜像或源代码安装，然后运行 mizuoto webui。",
        },
        {
          title: "安装引擎与模型",
          description: "首次运行的引导会为引擎建立独立的 Python 环境（含 PyTorch），再下载并校验一个模型。",
        },
        {
          title: "合成",
          description: "选择模型和音色，输入文本，边生成边听；结果连同参数一起保存在历史中。",
        },
        {
          title: "做成自己的",
          description: "把参考音频存进音色库，用项目朗读长篇作品，或用自己的录音微调模型。",
        },
      ],
    },
    highlights: {
      title: "引擎各不相同，用法始终一致",
      description: "保留每个引擎的上游代码，把环境、下载、排队与格式的差异交给 Mizuoto 处理。",
      items: [
        {
          title: "一种请求，七个引擎",
          description: "无论由哪个引擎处理，请求、音色、参数、任务与结果的形式都相同；不支持的选项在任务开始前就会指出。",
        },
        {
          title: "每个引擎一个独立环境",
          description: "Mizuoto 为每个引擎自动建立 Python 环境并安装匹配显卡的 PyTorch，引擎之间的依赖互不冲突；Mizuoto 本身从不导入 torch。",
        },
        {
          title: "网页、命令行与接口同源",
          description: "三者调用同一套核心操作，命令行的 --json 输出与 API 相同；另有兼容 OpenAI 的 /v1/audio/speech。",
        },
        {
          title: "边生成边播放",
          description: "通过 WebSocket 或分块 HTTP 流式传输音频，CosyVoice 还能边接收文本边合成；完整结果同样保存下来。",
        },
        {
          title: "引擎可以放在另一台机器",
          description: "在有显卡的机器上运行 mizuoto agent，本机的 Mizuoto 就能把那里的引擎当作自己的来用。",
        },
      ],
    },
    install: {
      title: "按文档完成安装",
      description:
        "Mizuoto 可以从 PyPI 安装，也可以使用 Docker 镜像或源代码。安装文档会说明系统要求、适合本机显卡的 torch 与每种方式的完整步骤；Fish Speech 与 IndexTTS 需先阅读并接受各自的许可证。",
      link: "查看安装文档",
      href: "/docs/installation#安装方式",
    },
    footer: {
      label: "页脚导航",
      docs: "文档",
      quickStart: "快速开始",
      faq: "常见问题",
      repository: "源代码",
      socialLabel: "社交链接",
      social: { blog: "博客", github: "GitHub", bilibili: "哔哩哔哩" },
    },
    preview: {
      label: "Mizuoto 合成界面示意",
      title: "合成",
      model: "模型",
      voice: "音色",
      text: "文本",
      textValue: "清晨的森林里，雾气还没有散去。远处传来一声鸟鸣，像是在回答谁的问候。",
      segments: "2 段",
      stream: "流式",
      speak: "合成",
      results: "结果",
      resultDetail: "24 kHz · WAV · 0:07",
      play: "播放",
    },
  },
} as const;

type TranslationShape<T> = T extends string
  ? string
  : T extends readonly unknown[]
    ? { readonly [K in keyof T]: TranslationShape<T[K]> }
    : T extends object
      ? { readonly [K in keyof T]: TranslationShape<T[K]> }
      : T;

export type SiteCopy = TranslationShape<typeof zhCN>;

const en = {
  pageRailLabel: "Page sections",
  theme: {
    hint: "Click to switch theme",
    modes: { system: "Theme: follow system", light: "Theme: light", dark: "Theme: dark" },
  },
  home: {
    sections: ["Home", "Features", "Workflow", "Highlights", "Install"],
    hero: {
      eyebrow: "One platform for open text-to-speech engines",
      title: "Seven speech engines,",
      titleAccent: "one platform",
      description:
        "Mizuoto puts CosyVoice, F5-TTS, GPT-SoVITS, OmniVoice, Qwen3-TTS, Fish Speech and IndexTTS behind one web UI, one command line and one HTTP API: speak, clone and design voices, convert and edit speech, read long-form projects and fine-tune models on your own recordings.",
      start: "Get started",
      learn: "Explore ",
      platformsLabel: "Supported platforms",
    },
    features: {
      title: "From one sentence to a whole book",
      description:
        "Every screen has its own tutorial, and everything it does can also be done from the command line and the HTTP API.",
      open: "Read the tutorial",
      items: [
        {
          title: "Speak",
          description:
            "Any installed model with a voice from your library, a clip to clone, a preset speaker or a description; style instructions and emotions where the model has them, and playback as the audio is made.",
          href: "/docs/guide-speak",
        },
        {
          title: "Projects",
          description:
            "Long-form work in chapters, read by a cast of speakers, each with its own model and voice; rendered line by line and exported with subtitles.",
          href: "/docs/guide-projects",
        },
        {
          title: "Tools",
          description:
            "Voice conversion (CosyVoice) says a recording in another voice with its words and timing; speech editing (F5-TTS) rewrites parts of it in the same voice.",
          href: "/docs/guide-tools",
        },
        {
          title: "Voices",
          description:
            "Clips with transcripts (speech recognition fills them in), designed voices, preset speakers and voices with their own fine-tuned weights, in one library.",
          href: "/docs/guide-voices",
        },
        {
          title: "Train",
          description:
            "Datasets from your recordings, then each engine's own fine-tuning recipe stage by stage, with live losses and spoken previews; the result is a voice's own weights or a new model.",
          href: "/docs/guide-train",
        },
      ],
    },
    workflow: {
      title: "From installation to your first words",
      description:
        "Four steps through the whole workflow, each with a web UI and a command-line way to do it.",
      steps: [
        {
          title: "Install Mizuoto",
          description: "From PyPI, the Docker image or a source checkout; then run mizuoto webui.",
        },
        {
          title: "Add an engine and a model",
          description:
            "The first-run guide builds the engine its own Python environment, PyTorch included, then downloads and verifies a model.",
        },
        {
          title: "Speak",
          description:
            "Pick a model and a voice, type the text and hear it as it is made; every result is kept with its settings.",
        },
        {
          title: "Make it yours",
          description:
            "Keep clips in the voice library, read a book with a project, or fine-tune a model on your own recordings.",
        },
      ],
    },
    highlights: {
      title: "Different engines, one way of working",
      description:
        "Each engine keeps its upstream code; Mizuoto takes care of environments, downloads, queues and formats.",
      items: [
        {
          title: "One request, seven engines",
          description:
            "Requests, voices, parameters, jobs and results look the same whichever engine serves them; what a model cannot do is reported before any job starts.",
        },
        {
          title: "An environment of its own for each engine",
          description:
            "Mizuoto builds each engine a Python environment with the PyTorch build for your graphics card, so their dependencies never clash; Mizuoto itself never imports torch.",
        },
        {
          title: "Web UI, command line and API from one core",
          description:
            "All three call the same operations, and the command line's --json output equals the API's; an OpenAI-compatible /v1/audio/speech comes with it.",
        },
        {
          title: "Hear it as it is made",
          description:
            "Audio streams over a WebSocket or chunked HTTP, and CosyVoice can even take the text as it is typed; the whole result is stored too.",
        },
        {
          title: "Engines on another machine",
          description:
            "Run mizuoto agent on the computer with the GPU, and the Mizuoto on yours uses its engines as its own.",
        },
      ],
    },
    install: {
      title: "Install with the guide",
      description:
        "Install Mizuoto from PyPI, or use the Docker image or a source checkout. The installation guide covers the requirements, the torch build for your graphics card and every way step by step; Fish Speech and IndexTTS need their licences read and accepted first.",
      link: "Read the installation guide",
      href: "/docs/installation#ways-to-install",
    },
    footer: {
      label: "Footer navigation",
      docs: "Docs",
      quickStart: "Quick start",
      faq: "FAQ",
      repository: "Source code",
      socialLabel: "Social links",
      social: { blog: "Blog", github: "GitHub", bilibili: "Bilibili" },
    },
    preview: {
      label: "Sketch of Mizuoto's Speak screen",
      title: "Speak",
      model: "Model",
      voice: "Voice",
      text: "Text",
      textValue:
        "Morning mist still hung between the trees. Somewhere far off a bird called, as if answering someone's greeting.",
      segments: "2 segments",
      stream: "Stream",
      speak: "Speak",
      results: "Results",
      resultDetail: "24 kHz · WAV · 0:07",
      play: "Play",
    },
  },
} as const satisfies SiteCopy;

const SITE_COPY: Record<Locale, SiteCopy> = { "zh-CN": zhCN, en };

export function getSiteCopy(locale: Locale): SiteCopy {
  return SITE_COPY[locale];
}

export function useSiteLocale(): Locale {
  const { locale } = useI18n();
  return isLocale(locale) ? locale : DEFAULT_LOCALE;
}

export function useSiteCopy(): SiteCopy {
  return getSiteCopy(useSiteLocale());
}
