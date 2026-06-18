# Veri Seti Formatı

Eğitim verisi **JSONL** (her satır bir JSON nesnesi). `backend/datasets/*.jsonl`.

## Şema

```jsonc
{
  "instruction": "Create a modern SaaS landing page hero section.",
  "input": "Style: premium, dark, orange accent #c45a26, responsive.",
  "output": "@@HTML@@\n<!DOCTYPE html>...\n@@CSS@@\n.btn{...}\n@@NOTES@@\nKısa açıklama\n@@END@@"
}
```

> **Neden `@@HTML@@` işaretçileri?** Eski `<html>...</html>` bölüm etiketleri
> gerçek HTML belgesinin kendi `<html>` etiketiyle çakışıyordu (parser yanlış
> bölümü alıyordu). `@@HTML@@/@@CSS@@/@@NOTES@@/@@END@@` HTML/CSS içeriğinde asla
> geçmez, bu yüzden ayrıştırma kesin ve kesilmeye dayanıklıdır.

- **instruction** (zorunlu): ne istediğin (doğal dil).
- **input** (opsiyonel): stil/kısıt detayları.
- **output** (tercih edilen): modelin üretmesini istediğin **yapısal** çıktı
  (`<response>` bloğu).

### Alternatif: html/css/notes alanları
`output` yerine ayrı alanlar verebilirsin; sistem `<response>` bloğunu otomatik kurar:
```jsonc
{ "instruction": "...", "input": "...",
  "html": "<!DOCTYPE html>...", "css": ".btn{...}", "notes": "Kısa açıklama" }
```

## Model giriş/çıkış (eğitimde nasıl paketlenir)

```
<|user|>
Instruction: {instruction}
Style: {input}
<|assistant|>
@@HTML@@
<!DOCTYPE html> ... (tam belge, stil hariç)
@@CSS@@
... css ...
@@NOTES@@
... kısa açıklama ...
@@END@@
<|end|>
```
(`<|user|>`, `<|assistant|>`, `<|end|>`, `<|endoftext|>` özel token; `@@HTML@@`
vb. işaretçiler **düz metin** — model bunları öğrenir. Üretim `@@END@@` ya da
`<|end|>` görülünce durur.)

## 8 kategori

1. Prompt → HTML/CSS
2. Prompt → Bootstrap layout
3. Bozuk CSS → düzeltilmiş CSS
4. UI tarifi → bileşen kodu
5. Renk paleti → CSS değişkenleri
6. Masaüstü layout → responsive layout
7. Basit bileşen üretimi
8. Tek-dosya tam HTML sayfası

`backend/datasets/build_seed.py` bu 8 kategoride parametrik örnekler üretir;
`backend/datasets/seed.jsonl` çıktısını verir.

## Kalite notu

Sıfırdan model **çok veri** ister. Birkaç yüz örnek boru hattını test eder ama
tutarlı çıktı için **binlerce** iyi örnek + uzun eğitim gerekir. Veriyi
çoğaltmak en yüksek etkili adımdır.
