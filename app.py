from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image, ImageEnhance, ImageFilter
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SINGLE_STAFF_PROMPT = """This image shows ONE line of sheet music with solfege syllables written below each note.

Read the solfege syllables below the notes from LEFT to RIGHT. They look like: do, re, mi, fa, sol, la, si, sib, fa#, es, etc.

Rules:
- "es" = Mib (E flat)
- "sib" = Sib (B flat)  
- "fa#" = Fa# (F sharp)
- Determine duration from note appearance (quarter=1, eighth=0.5, half=2, whole=4, dotted quarter=1.5)
- Key signature at the start applies to all notes

Output format - write ONLY this one line, nothing else:
PORTE 1: Do4(1) Re4(0.5) Mib4(0.5) Sol4(1) ..."""


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def enhance_image(img):
    w, h = img.size
    if w < 1200:
        scale = 1200 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.convert('L').convert('RGB')
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def ask_gpt_single(b64, staff_num):
    """Tek bir porte için GPT'ye sor."""
    prompt = SINGLE_STAFF_PROMPT.replace('PORTE 1', f'PORTE {staff_num}')
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        },
        timeout=60
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result:
        return None, "Empty response"
    return result["choices"][0]["message"]["content"].strip(), None


def parse_notes(text, staff_idx):
    notes = []
    pitch_map = {
        'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa',
        'sol': 'Sol', 'la': 'La', 'si': 'Si'
    }
    # PORTE satırını bul
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line.upper().startswith('PORTE'):
            continue
        colon = line.find(':')
        if colon < 0:
            continue
        notes_part = line[colon + 1:].strip()
        for m in re.finditer(r'([A-Za-z]+)(\d)([#b]?)\(([0-9.]+)\)', notes_part):
            raw_pitch = m.group(1).lower()
            pitch = pitch_map.get(raw_pitch)
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
        if notes:
            break
    return notes


def crop_staves(img, staff_count):
    """Görüntüyü porte sayısına göre yatay şeritler halinde böl."""
    w, h = img.size
    strip_h = h // staff_count
    crops = []
    for i in range(staff_count):
        top = i * strip_h
        bottom = (i + 1) * strip_h if i < staff_count - 1 else h
        # Biraz padding ekle
        pad = int(strip_h * 0.05)
        top = max(0, top - pad)
        bottom = min(h, bottom + pad)
        crops.append(img.crop((0, top, w, bottom)))
    return crops


def detect_staff_count(img):
    w, h = img.size
    ratio = h / w
    if ratio < 0.35:
        return 1
    elif ratio < 0.6:
        return 2
    elif ratio < 0.85:
        return 3
    elif ratio < 1.1:
        return 4
    else:
        return 5


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Goruntu eksik"}), 400

        b64 = data['image']
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        staff_count = detect_staff_count(img)

        # Her porteyi ayrı crop et
        crops = crop_staves(img, staff_count)

        all_notes = []
        all_staves = []
        raw_parts = []

        for i, crop in enumerate(crops):
            enhanced = enhance_image(crop)
            b64_crop = image_to_b64(enhanced)
            text, err = ask_gpt_single(b64_crop, i + 1)

            if err or not text:
                continue

            raw_parts.append(f"PORTE {i+1}: {text}")
            notes = parse_notes(text, i)

            if notes:
                all_notes.extend(notes)
                all_staves.append({"index": i, "noteCount": len(notes)})

        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadi.",
                "debug_raw": raw_parts
            })

        actual_staff_count = len(set(n.get("staffIndex", 0) for n in all_notes))

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": all_staves,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": actual_staff_count,
                    "timeSignature": "4/4",
                    "clef": "treble",
                    "rawText": '\n'.join(raw_parts)
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    return jsonify({"success": False, "error": "PDF ozelligi yakinda eklenecek."}), 501


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
