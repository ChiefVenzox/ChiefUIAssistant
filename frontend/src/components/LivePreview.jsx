import { useMemo } from "react";
import { buildPreviewDoc } from "../utils.js";

// Üretilen HTML/CSS'i canlı bir iframe'de gösterir. sandbox ile izole.
export default function LivePreview({ html, css }) {
  const doc = useMemo(() => buildPreviewDoc(html, css), [html, css]);
  if (!html && !css) {
    return <div className="preview-empty">Önizleme burada görünecek…</div>;
  }
  return (
    <iframe
      className="preview-frame"
      title="preview"
      sandbox="allow-scripts"
      srcDoc={doc}
    />
  );
}
