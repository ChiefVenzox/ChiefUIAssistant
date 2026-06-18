import { useState } from "react";
import LivePreview from "./LivePreview.jsx";

const TABS = ["Önizleme", "HTML", "CSS", "Notlar"];

function CodeBlock({ code }) {
  const copy = () => navigator.clipboard?.writeText(code || "");
  return (
    <div className="code-wrap">
      <button className="copy-btn" onClick={copy}>
        Kopyala
      </button>
      <pre className="code-block">
        <code>{code || "(boş)"}</code>
      </pre>
    </div>
  );
}

export default function CodeTabs({ html, css, notes }) {
  const [tab, setTab] = useState(0);
  return (
    <div className="tabs">
      <div className="tab-bar">
        {TABS.map((t, i) => (
          <button
            key={t}
            className={"tab" + (i === tab ? " active" : "")}
            onClick={() => setTab(i)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="tab-body">
        {tab === 0 && <LivePreview html={html} css={css} />}
        {tab === 1 && <CodeBlock code={html} />}
        {tab === 2 && <CodeBlock code={css} />}
        {tab === 3 && <CodeBlock code={notes} />}
      </div>
    </div>
  );
}
