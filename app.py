from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
import io
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ─── PROMPT ───────────────────────────────────────────────────────────────────

DETECT_PROMPT = """This image has rows of horizontal lines with text labels below them.

Count the rows and estimate where each row starts and ends as a percentage of image height.
0% = top, 100% = bottom. Include generous padding.

Reply ONLY in this exact format, nothing else:
COUNT: 4
ROW1: 2 27
ROW2: 25 52
ROW3: 50 77
ROW4: 75 100"""

def read_prompt(n):
    return f"""This image shows ONE row of horizontal lines with small text labels written below.

Read ALL text labels below the lines from left to right. Do not skip any label.

Output ONLY this one line, nothing else:
PORTE {n}: Do4(1) Re4(0.5) Mib4(0.5) ...

Conversion rules:
- "es" or "eb" → Mib4(0.5)
- "sib" → Sib4(1)
- "fa#" or "faş" or "fa♯" → Fa4#(1)
- "re#" or "reş" or "re♯" → Re4#(1)
- "sol#" or "sol♯" → Sol4#(1)
- "do#" or "do♯" → Do4#(1)
- "la#" or "la♯" → La4#(1)
- All others: capitalize first letter, add octave 4, duration 1
- Eighth note (with flag/beam) = 0.5
- Quarter note = 1
- Half note = 2
- Whole note = 4
- Dotted quarter = 1.5
- Dotted half = 3"""

# ─── GÖRÜNTÜ İYİLEŞTİRME ────────────────────────────────────────────────────

def preprocess_image(img_pil):
    """
    PIL görüntüsünü OpenCV ile iyileştir:
    1. Gri tonlamaya çevir
    2. Gürültüyü azalt
    3. Kontrast artır (CLAHE)
    4. Adaptif threshold
    5. Eğimi düzelt (deskew)
    """
    # PIL → numpy
    img_np = np.array(img_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Griye çevir
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Gürültü azalt
    denoised = cv2.medianBlur(gray, 3)

    # 3. CLAHE ile kontrast iyileştir
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Adaptif threshold (siyah-beyaz)
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=8
    )

    # 5. Eğim tespiti ve düzeltme
    binary_inv = cv2.bitwise_not(binary)
    coords = np.column_stack(np.where(binary_inv > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:  # 0.5 dereceden fazla eğim varsa düzelt
            (h, w) = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    # numpy → PIL
    result = Image.fromarray(binary).convert('RGB')
    return result


def enhance_crop(img_pil):
    """Crop edilmiş tek bir porteyi GPT için iyileştir."""
    img_np = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Boyutu büyüt (minimum 800px genişlik)
    h, w = gray.shape
    if w < 800:
        scale = 800 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    # Kontrast artır
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Keskinleştir
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return Image.fromarray(sharpened).convert('RGB')


# ─── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────

def pil_to_b64(img_pil, quality=92):
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def ask_gpt(b64, prompt, max_tokens=512):
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "max_tokens": max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=90
        )
        result = resp.json()
        if "error" in result:
            return None, result["error"].get("message", "API hatası")
        if "choices" not in result or not result["choices"]:
            return None, "Boş yanıt"
        return result["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        return None, str(e)


def parse_notes(text, staff_idx):
    notes = []
    pitch_map = {
        'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa',
        'sol': 'Sol', 'la': 'La', 'si': 'Si'
    }
    for m in re.finditer(r'([A-Za-z]+)(\d)([#b]?)\(([0-9.]+)\)', text):
        pitch = pitch_map.get(m.group(1).lower())
        if not pitch:
            continue
        notes.append({
            "pitch": pitch,
            "octave": int(m.group(2)),
            "duration": float(m.group(4)),
            "accidental": m.group(3) if m.group(3) else None,
            "staffIndex": staff_idx,
            "confidence": 0.9
        })
    return notes


def crop_staff(img_pil, top_pct, bottom_pct):
    w, h = img_pil.size
    top = max(0, int((top_pct / 100) * h))
    bottom = min(h, int((bottom_pct / 100) * h))
    return img_pil.crop((0, top, w, bottom))


# ─── ENDPOINT'LER ────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "opencv": cv2.__version__})


@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Görüntü eksik"}), 400

        # Görüntüyü yükle
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Boyutu sınırla
        w, h = img.size
        max_dim = 2500
        if w > max_dim or h > max_dim:
            scale = min(max_dim / w, max_dim / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # ── ADIM 1: OpenCV ile görüntü iyileştir ──
        img_clean = preprocess_image(img)
        full_b64 = pil_to_b64(img_clean)

        # ── ADIM 2: Porte konumlarını tespit et ──
        detect_text, err = ask_gpt(full_b64, DETECT_PROMPT, max_tokens=256)
        if err or not detect_text:
            return jsonify({"success": False, "error": "Porte tespiti başarısız: " + (err or "boş yanıt")})

        count_match = re.search(r'COUNT:\s*(\d+)', detect_text)
        count = int(count_match.group(1)) if count_match else 4
        count = min(count, 10)  # maksimum 10 porte

        rows = []
        for i in range(1, count + 1):
            row_match = re.search(rf'ROW{i}:\s*(\d+)\s+(\d+)', detect_text)
            if row_match:
                rows.append({'top': int(row_match.group(1)), 'bottom': int(row_match.group(2))})
            else:
                rows.append({
                    'top': round((i - 1) * 100 / count),
                    'bottom': round(i * 100 / count)
                })

        # ── ADIM 3: Her porteyi crop et + iyileştir + GPT'ye gönder ──
        all_notes = []
        all_staves = []
        raw_parts = []
        errors = []

        for i, row in enumerate(rows):
            try:
                # Gerçek crop
                cropped = crop_staff(img, row['top'], row['bottom'])

                # Crop'u iyileştir
                cropped_enhanced = enhance_crop(cropped)
                cropped_b64 = pil_to_b64(cropped_enhanced, quality=95)

                # GPT'ye gönder
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
                    all_staves.append({"index": i, "noteCount": len(notes)})
                else:
                    errors.append(f"Porte {i+1}: nota parse edilemedi ({row_text[:80]})")

            except Exception as e:
                errors.append(f"Porte {i+1} hatası: {str(e)}")

        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadı.",
                "debug": {
                    "detect": detect_text,
                    "rows": rows,
                    "raw": raw_parts,
                    "errors": errors
                }
            })

        staff_count = len(set(n.get("staffIndex", 0) for n in all_notes))

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
        return jsonify({"success": False, "error": "Sunucu hatası: " + str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    return jsonify({"success": False, "error": "PDF özelliği yakında eklenecek."}), 501


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
