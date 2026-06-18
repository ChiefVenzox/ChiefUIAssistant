// Backend ile iletişim. Vite proxy sayesinde relative URL kullanılır (/api, /ws).

export async function checkHealth() {
  const r = await fetch("/api/health");
  return r.json();
}

export async function getHistory(limit = 20) {
  const r = await fetch(`/api/history?limit=${limit}`);
  return r.json();
}

// REST (fallback / streaming yoksa)
export async function generateRest(req) {
  const r = await fetch("/api/generate-ui", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error((await r.json()).detail || "İstek başarısız");
  return r.json();
}

// WebSocket streaming. onToken(text), onDone(result), onError(msg).
// Geriye, üretimi iptal etmek için bir close() fonksiyonu döner.
export function streamGenerate(req, { onToken, onDone, onError }) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
  let settled = false; // done/error/iptal alındı mı?

  ws.onopen = () => ws.send(JSON.stringify(req));
  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    if (msg.type === "token") onToken(msg.text);
    else if (msg.type === "done") {
      settled = true;
      onDone(msg);
    } else if (msg.type === "error") {
      settled = true;
      onError(msg.message);
    }
  };
  ws.onerror = () => {
    if (!settled) {
      settled = true;
      onError("WebSocket bağlantı hatası");
    }
  };
  // Sunucu done/error göndermeden kapanırsa (kopma/çökme) takılı kalmayı önle
  ws.onclose = () => {
    if (!settled) {
      settled = true;
      onError("Bağlantı beklenmedik şekilde kapandı");
    }
  };

  return () => {
    settled = true; // kullanıcı iptali -> onclose gereksiz hata göstermesin
    try {
      ws.close();
    } catch {
      /* yoksay */
    }
  };
}
