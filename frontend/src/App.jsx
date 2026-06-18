import { useEffect, useRef, useState } from "react";
import ThreeBackground from "./components/ThreeBackground.jsx";
import CodeTabs from "./components/CodeTabs.jsx";
import { checkHealth, getHistory, streamGenerate } from "./api.js";
import { exportZip } from "./utils.js";

// Backend'in parse_response'unun istemci tarafı eşi (canlı önizleme için).
// @@HTML@@ / @@CSS@@ / @@NOTES@@ işaretçileri arası; kesilmeye dayanıklı.
function jsParse(raw) {
  const grab = (name) => {
    const re = new RegExp(`@@${name}@@([\\s\\S]*?)(?:@@HTML@@|@@CSS@@|@@NOTES@@|@@END@@|$)`, "i");
    const m = raw.match(re);
    return m ? m[1].trim() : "";
  };
  let html = grab("HTML");
  const css = grab("CSS");
  const notes = grab("NOTES");
  if (!html && !css) html = raw.replace(/@@(HTML|CSS|NOTES|END)@@/g, "").trim();
  return { html, css, notes };
}

export default function App() {
  const [instruction, setInstruction] = useState(
    "Create a modern dark landing page hero section"
  );
  const [input, setInput] = useState(
    "Style: dark, orange accent #c45a26, responsive"
  );
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(512);

  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState({ html: "", css: "", notes: "" });
  const [validation, setValidation] = useState(null);
  const [history, setHistory] = useState([]);
  const cancelRef = useRef(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ ready: false }));
    getHistory().then((d) => setHistory(d.items || [])).catch(() => {});
  }, []);

  const refreshHistory = () =>
    getHistory().then((d) => setHistory(d.items || [])).catch(() => {});

  const onGenerate = () => {
    if (busy) {
      cancelRef.current?.(); // iptal
      setBusy(false);
      return;
    }
    setBusy(true);
    setValidation(null);
    setResult({ html: "", css: "", notes: "" });
    let raw = "";
    cancelRef.current = streamGenerate(
      { instruction, input, temperature, max_new_tokens: maxTokens, top_k: 40, top_p: 0.95 },
      {
        onToken: (t) => {
          raw += t;
          setResult(jsParse(raw)); // canlı güncelle
        },
        onDone: (msg) => {
          setResult({ html: msg.html, css: msg.css, notes: msg.notes });
          setValidation(msg.validation);
          setBusy(false);
          refreshHistory();
        },
        onError: (m) => {
          setValidation({ ok: false, issues: [m], warnings: [] });
          setBusy(false);
        },
      }
    );
  };

  const ready = health?.ready;

  return (
    <div className="app">
      <ThreeBackground />

      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" /> ChiefUI <small>Assistant</small>
        </div>
        <div className="status">
          {ready
            ? `hazır · ${Math.round(health.params_m)}M · ${health.device}`
            : "model yüklenmedi (önce eğit)"}
        </div>
      </header>

      <main className="layout">
        {/* Sol panel: prompt + ayarlar + geçmiş */}
        <section className="panel left">
          <label className="field">
            <span>Talimat (instruction)</span>
            <textarea
              rows={3}
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="Örn: Create a responsive pricing card"
            />
          </label>
          <label className="field">
            <span>Stil / detay (input)</span>
            <textarea
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Örn: Style: dark, orange #c45a26, rounded"
            />
          </label>

          <div className="controls">
            <label>
              sıcaklık {temperature.toFixed(2)}
              <input
                type="range" min="0" max="1.5" step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(+e.target.value)}
              />
            </label>
            <label>
              max token {maxTokens}
              <input
                type="range" min="64" max="1024" step="64"
                value={maxTokens}
                onChange={(e) => setMaxTokens(+e.target.value)}
              />
            </label>
          </div>

          <div className="btn-row">
            <button className="primary" onClick={onGenerate} disabled={!ready && !busy}>
              {busy ? "Durdur" : "Üret"}
            </button>
            <button
              className="ghost"
              onClick={() => exportZip(result)}
              disabled={!result.html && !result.css}
            >
              ZIP indir
            </button>
          </div>

          {validation && (
            <div className={"validation " + (validation.ok ? "ok" : "bad")}>
              <strong>{validation.ok ? "✓ Doğrulama geçti" : "⚠ Sorunlar"}</strong>
              {validation.issues?.map((i, k) => <div key={"i" + k}>• {i}</div>)}
              {validation.warnings?.map((w, k) => (
                <div key={"w" + k} className="warn">○ {w}</div>
              ))}
            </div>
          )}

          <div className="history">
            <div className="history-title">Geçmiş</div>
            {history.length === 0 && <div className="muted">Henüz yok.</div>}
            {history.map((h) => (
              <button
                key={h.id}
                className="history-item"
                onClick={() =>
                  setResult({ html: h.html, css: h.css, notes: h.notes })
                }
                title={h.instruction}
              >
                #{h.id} {h.instruction?.slice(0, 40)}
              </button>
            ))}
          </div>
        </section>

        {/* Sağ panel: önizleme + kod sekmeleri */}
        <section className="panel right">
          <CodeTabs html={result.html} css={result.css} notes={result.notes} />
        </section>
      </main>
    </div>
  );
}
