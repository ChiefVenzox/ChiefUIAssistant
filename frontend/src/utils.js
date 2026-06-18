import JSZip from "jszip";

// HTML + CSS'i iframe önizlemesi için tek bir belgeye birleştirir.
export function buildPreviewDoc(html, css) {
  html = html || "";
  css = css || "";
  const styleTag = css.trim() ? `<style>\n${css}\n</style>` : "";

  // Tam belge mi? <head> varsa style'ı oraya enjekte et.
  if (/<html[\s>]/i.test(html)) {
    if (/<\/head>/i.test(html)) {
      return html.replace(/<\/head>/i, `${styleTag}\n</head>`);
    }
    return html.replace(/<html[^>]*>/i, (m) => `${m}\n<head>${styleTag}</head>`);
  }
  // Parça ise tam belgeye sar.
  return `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
${styleTag}
</head><body>
${html}
</body></html>`;
}

// Üretilen kodu ZIP olarak indir.
export async function exportZip({ html, css, notes }) {
  const zip = new JSZip();
  const hasFullDoc = /<html[\s>]/i.test(html || "");
  if (hasFullDoc) {
    zip.file("index.html", html || "");
    if ((css || "").trim()) zip.file("styles.css", css);
  } else {
    // parça -> index.html (styles.css'e link) + styles.css
    zip.file(
      "index.html",
      `<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="styles.css">
</head><body>
${html || ""}
</body></html>`
    );
    zip.file("styles.css", css || "");
  }
  if ((notes || "").trim()) zip.file("notes.txt", notes);

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chiefui-export.zip";
  a.click();
  URL.revokeObjectURL(url);
}
