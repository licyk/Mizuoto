import { ChevronDown, Play, Radio, Speech, UserRound } from "lucide-react";

import { useSiteCopy } from "@/lib/site-copy";

/** What the sketch shows: a real catalogue model and a designed library voice. */
const PREVIEW_MODEL = "Qwen3-TTS 1.7B CustomVoice";
const PREVIEW_VOICE = "Narrator";

/**
 * Bar heights (percent) of the result's waveform: two phrases with a pause
 * between them, fixed so the prerendered page and the client agree.
 */
const WAVEFORM = Array.from({ length: 56 }, (_, index) => {
  if (index >= 25 && index <= 28) return 8;
  const envelope = Math.sin(((index % 28) / 28) * Math.PI);
  const ripple = 0.55 + 0.45 * Math.abs(Math.sin(index * 1.7) * Math.cos(index * 0.6));
  return Math.round(12 + 82 * envelope * ripple);
});

/** Mizuoto's mark, as the web UI and the favicon draw it. */
function MizuotoMark() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 12h2" />
      <path d="M7 8v8" />
      <path d="M11 4v16" />
      <path d="M15 7v10" />
      <path d="M19 10v4" />
      <path d="M21 12h0" />
    </svg>
  );
}

/** A static sketch of the Speak screen: model, voice, text, the Speak button and a result. */
export default function ProductWorkspacePreview() {
  const copy = useSiteCopy().home.preview;

  return (
    <figure className="product-preview page-interactive" aria-label={copy.label}>
      <div className="product-preview-window page-interactive-part">
        <div className="product-preview-titlebar">
          <div className="product-preview-window-controls" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="product-preview-brand">
            <span className="product-preview-brand-mark" aria-hidden="true">
              <MizuotoMark />
            </span>
            Mizuoto
          </div>
          <div className="product-preview-titlebar-title">{copy.title}</div>
        </div>

        <div className="product-preview-body">
          <div className="product-preview-fields">
            <div className="product-preview-field">
              <span>{copy.model}</span>
              <strong>
                {PREVIEW_MODEL}
                <ChevronDown aria-hidden="true" />
              </strong>
            </div>
            <div className="product-preview-field">
              <span>{copy.voice}</span>
              <strong>
                <UserRound aria-hidden="true" />
                {PREVIEW_VOICE}
                <ChevronDown aria-hidden="true" />
              </strong>
            </div>
          </div>

          <div className="product-preview-text">
            <span>{copy.text}</span>
            <p>{copy.textValue}</p>
            <small>{copy.segments}</small>
          </div>

          <div className="product-preview-actions">
            <span className="product-preview-chip">
              <Radio aria-hidden="true" />
              {copy.stream}
            </span>
            <span className="product-preview-speak">
              <Speech aria-hidden="true" />
              {copy.speak}
            </span>
          </div>

          <div className="product-preview-result">
            <span className="product-preview-play" aria-label={copy.play} role="img">
              <Play aria-hidden="true" />
            </span>
            <div className="product-preview-result-body">
              <div className="product-preview-result-heading">
                <strong>
                  {copy.results} · {PREVIEW_VOICE}
                </strong>
                <small>{copy.resultDetail}</small>
              </div>
              <div className="product-preview-wave" aria-hidden="true">
                {WAVEFORM.map((height, index) => (
                  <span key={index} style={{ height: `${height}%` }} />
                ))}
                <span className="product-preview-playhead" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </figure>
  );
}
