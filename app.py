from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image, ImageFilter, ImageOps
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# PROMPT'LAR
# ─────────────────────────────────────────────

DETECT_PROMPT = """This is a sheet music image. Your task is to count the number of staff systems (groups of 5 horizontal lines) visible.

Look carefully at the image and identify each distinct staff (set of 5 lines).

Reply ONLY in this exact format, no extra text:
COUNT: 4
ROW1: 5 28
ROW2: 28 52
ROW3: 52 76
ROW4: 76 100

Where ROW numbers are percentages from top (0) to bottom (100) of image height.
Include some padding: start a few percent before the first line, end a few percent after the last line."""

def read_prompt(n):
    return f"""You are an expert music notation reader. This image shows ONE staff (5 horizontal lines) with musical notes on or between the lines.

Your task: identify every note from LEFT to RIGHT and output them in Turkish solfege notation.

TREBLE CLEF note positions (lines from bottom to top: Mi4, Sol4, Si4, Re5, Fa5):
- Notes ON lines (bottom to top): Mi4, Sol4, Si4, Re5, Fa5
- Notes IN spaces (bottom to top): Fa4, La4, Do5, Mi5
- Notes BELOW staff: Re4 (just below), Do4 (ledger line below)
- Notes ABOVE staff: Sol5 (just above), La5, etc.

NOTE VALUES:
- Whole note (open oval, no stem) = 4
- Half note (open oval with stem) = 2
- Quarter note (filled oval with stem) = 1
- Eighth note (filled oval, stem with flag/beam) = 0.5
- Dotted quarter = 1.5
- Dotted half = 3

ACCIDENTALS:
- Sharp (#) before note: add # after note name, e.g. Fa4#
- Flat (b) before note: add b after note name, e.g. Sib4
- Natural cancels previous accidental

OUTPUT FORMAT - Reply ONLY with this line, nothing else:
PORTE {n}: Do4(1) Re4(0.5) Mi4(1) ...

IMPORTANT:
- List every single note, do not skip any
- If unsure between two notes, pick the most likely one
- Do NOT include rests, clef symbols, time signatures, or barlines
- Do NOT add any explanation, just the PORTE line"""


# ─────────────────────────────────────────────
# GÖRÜNTÜ ÖN İŞLEME
# ─────────────────────────────────────────────

def preprocess_image(img):
    """Görüntüyü OCR için iyileştir"""
    # Grayscale
    gray = img.convert('L')
    
    # Kontrast artır
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    
    # Sharpness
    enhancer2 = ImageEnhance.Sharpness(gray)
    gray = enhancer2.enhance(2.0)
    
    # Gürültü azalt
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    
    # Tekrar RGB'ye çevir (GPT için)
    return gray.convert('RGB')


def image_to_b64(img, quality=92):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# GPT ÇAĞRISI
# ─────────────────────────────────────────────

def ask_gpt(b64, prompt, max_tokens=2048):
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
            "max_tokens": max_tokens,
            "temperature": 0,  # Deterministik sonuç
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        },
        timeout=90
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result or not result["choices"]:
        return None, "Boş yanıt"
    return result["choices"][0]["message"]["content"].strip(), None


# ─────────────────────────────────────────────
# NOTA PARSE
# ─────────────────────────────────────────────

PITCH_NORMALIZE = {
    'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa',
    'sol': 'Sol', 'la': 'La', 'si': 'Si',
    # Yaygın alternatif yazımlar
    'ut': 'Do', 'ti': 'Si', 'te': 'Si',
    # GPT bazen İngilizce yazar
    'c': 'Do', 'd': 'Re', 'e': 'Mi', 'f': 'Fa',
    'g': 'Sol', 'a': 'La', 'b': 'Si',
}

def parse_notes(text, staff_idx):
    """GPT çıktısından nota listesi çıkar"""
    notes = []
    
    # PORTE satırını bul
    line = text
    if f'PORTE {staff_idx}:' in text:
        line = text.split(f'PORTE {staff_idx}:')[1].strip()
    elif 'PORTE' in text:
        # Herhangi bir PORTE satırı
        for t in text.split('\n'):
            if 'PORTE' in t and ':' in t:
                line = t.split(':', 1)[1].strip()
                break
    
    # Pattern: NotaAdı + Oktav + (isteğe bağlı #/b) + (süre)
    # Örnek: Do4(1), Sib4(0.5), Fa4#(1), Mib5(2)
    pattern = r'([A-Za-z]+)\s*(\d)\s*([#b♭♯]?)\s*\(([0-9.]+)\)'
    
    for m in re.finditer(pattern, line):
        raw_name = m.group(1).lower()
        octave_str = m.group(2)
        accidental = m.group(3)
        duration_str = m.group(4)
        
        # Nota adını normalize et
        pitch = PITCH_NORMALIZE.get(raw_name)
        if not pitch:
            # Türkçe özel: "mib" → "Mi" + "b", "sib" → "Si" + "b", "fa#" → "Fa" + "#"
            for key in sorted(PITCH_NORMALIZE.keys(), key=len, reverse=True):
                if raw_name.startswith(key):
                    pitch = PITCH_NORMALIZE[key]
                    suffix = raw_name[len(key):]
                    if suffix in ['b', '♭'] and not accidental:
                        accidental = 'b'
                    elif suffix in ['#', '♯'] and not accidental:
                        accidental = '#'
                    break
        
        if not pitch:
            continue
        
        try:
            octave = int(octave_str)
            duration = float(duration_str)
        except ValueError:
            continue
        
        # Makul aralık kontrolü
        if octave < 2 or octave > 7:
            continue
        if duration not in [0.25, 0.5, 1, 1.5, 2, 3, 4]:
            duration = round(duration * 4) / 4  # En yakın geçerli değere yuvarla
        
        notes.append({
            "pitch": pitch,
            "octave": octave,
            "duration": duration,
            "accidental": accidental if accidental else None,
            "staffIndex": staff_idx,
            "confidence": 0.9
        })
    
    return notes


# ─────────────────────────────────────────────
# PORTE KROP
# ─────────────────────────────────────────────

def crop_staff(img, top_pct, bottom_pct):
    w, h = img.size
    # Biraz padding ekle ama sınırı aşma
    padding = 1.0  # %1 ekstra
    top = max(0, int(((top_pct - padding) / 100) * h))
    bottom = min(h, int(((bottom_pct + padding) / 100) * h))
    return img.crop((0, top, w, bottom))


def parse_detect_response(text, fallback_count=4):
    """Porte tespit yanıtını parse et"""
    count_match = re.search(r'COUNT:\s*(\d+)', text)
    count = int(count_match.group(1)) if count_match else fallback_count
    count = max(1, min(count, 12))  # 1-12 arası sınırla
    
    rows = []
    for i in range(1, count + 1):
        row_match = re.search(rf'ROW{i}:\s*(\d+)\s+(\d+)', text)
        if row_match:
            top = int(row_match.group(1))
            bottom = int(row_match.group(2))
            # Sınır kontrolü
            top = max(0, min(top, 99))
            bottom = max(top + 5, min(bottom, 100))
            rows.append({'top': top, 'bottom': bottom})
        else:
            # Eşit böl
            rows.append({
                'top': round((i - 1) * 100 / count),
                'bottom': round(i * 100 / count)
            })
    
    return count, rows


# ─────────────────────────────────────────────
# ENDPOINT'LER
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "version": "2.0"})


@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Görüntü eksik"}), 400

        # Görüntüyü yükle
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Boyut sınırla (GPT için)
        w, h = img.size
        if w > 2400 or h > 3600:
            scale = min(2400 / w, 3600 / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Görüntüyü iyileştir
        enhanced = preprocess_image(img)
        full_b64 = image_to_b64(enhanced)

        # ── ADIM 1: Porte tespiti ──
        detect_text, err = ask_gpt(full_b64, DETECT_PROMPT, max_tokens=256)
        if err or not detect_text:
            return jsonify({
                "success": False,
                "error": "Porte tespiti başarısız: " + (err or "boş yanıt")
            })

        count, rows = parse_detect_response(detect_text)

        # ── ADIM 2: Her porteyi oku ──
        all_notes = []
        all_staves = []
        raw_parts = []
        errors = []

        for i, row in enumerate(rows):
            # Orjinal (işlenmemiş) görüntüden crop — daha iyi sonuç verir
            cropped = crop_staff(enhanced, row['top'], row['bottom'])
            
            # Crop çok küçükse atla
            cw, ch = cropped.size
            if ch < 30 or cw < 50:
                continue
            
            cropped_b64 = image_to_b64(cropped)
            row_text, err = ask_gpt(cropped_b64, read_prompt(i + 1), max_tokens=1024)
            
            if err:
                errors.append(f"Porte {i+1}: {err}")
                continue
            if not row_text:
                errors.append(f"Porte {i+1}: boş yanıt")
                continue

            raw_parts.append(row_text)
            notes = parse_notes(row_text, i)
            
            if notes:
                all_notes.extend(notes)
                all_staves.append({
                    "index": i,
                    "noteCount": len(notes),
                    "rawText": row_text
                })

        # Hiç nota bulunamadıysa
        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadı. Daha net bir fotoğraf deneyin.",
                "debug": {
                    "detectResponse": detect_text,
                    "rawParts": raw_parts,
                    "errors": errors
                }
            })

        staff_count = len(all_staves)

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": all_staves,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": staff_count,
                    "timeSignature": "4/4",
                    "clef": "treble",
                    "rawText": "\n".join(raw_parts)
                }
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": "Sunucu hatası: " + str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    return jsonify({
        "success": False,
        "error": "PDF özelliği yakında eklenecek."
    }), 501


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
