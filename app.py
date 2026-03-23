from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image, ImageEnhance, ImageFilter
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def build_prompt(staff_count):
    return f"""You are a music sheet reader. This image contains exactly {staff_count} staff lines with Turkish solfege syllables written below each note (do, re, mi, fa, sol, la, si, sib, fa#, es, etc.).

YOUR TASK: Read ALL notes from ALL {staff_count} staves, left to right, without skipping any.

READING PRIORITY:
1. Read the SOLFEGE TEXT written below the notes (most accurate source)
2. If no solfege text, determine pitch from note head position on staff

SOLFEGE MAPPINGS:
- "es" or "eb" = Mib (E flat)
- "sib" = Sib (B flat)
- "fa#" = Fa# (F sharp)
- "do#" = Do# (C sharp)
- "sol#" = Sol# (G sharp)

KEY SIGNATURE: Apply sharps/flats shown at the beginning to all relevant notes throughout.

DURATIONS:
- Whole note = 4
- Half note = 2
- Quarter note = 1
- Eighth note = 0.5
- Sixteenth note = 0.25
- Dotted quarter = 1.5
- Dotted half = 3

OUTPUT FORMAT - write ONLY this, nothing else:
PORTE 1: Do4(1) Re4(0.5) Mib4(0.5) Sol4(1) ...
PORTE 2: Sib4(0.5) Do5(1) Re5(0.5) ...
PORTE 3: ...
PORTE 4: ...

IMPORTANT: You MUST write all {staff_count} PORTE lines. Do not skip any."""


def detect_staff_count(img):
    w, h = img.size
    ratio = h / w
    if ratio < 0.4:
        return 1
    elif ratio < 0.7:
        return 2
    elif ratio < 1.0:
        return 3
    elif ratio < 1.4:
        return 4
    else:
        return 5


def enhance_image(img):
    """Görüntüyü GPT için iyileştir."""
    # Boyutu büyüt - minimum 1500px genişlik
    w, h = img.size
    if w < 1500:
        scale = 1500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Maksimum boyut sınırı
    w, h = img.size
    if w > 2500 or h > 2500:
        scale = min(2500 / w, 2500 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Griye çevir - nota kağıtları için daha net
    img = img.convert('L').convert('RGB')

    # Kontrast artır
    img = ImageEnhance.Contrast(img).enhance(2.0)

    # Keskinleştir
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def ask_gpt(b64, staff_count):
    prompt = build_prompt(staff_count)
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        },
        timeout=120
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result:
        return None, "Empty response"
    return result["choices"][0]["message"]["content"].strip(), None


def parse_notes(text):
    all_notes = []
    staves = []
    pitch_map = {
        'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa',
        'sol': 'Sol', 'la': 'La', 'si': 'Si'
    }

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line.upper().startswith('PORTE'):
            continue
        colon = line.find(':')
        if colon < 0:
            continue

        staff_num = re.search(r'\d+', line[:colon])
        staff_idx = (int(staff_num.group()) - 1) if staff_num else len(staves)
        notes_part = line[colon + 1:].strip()
        notes = []

        for m in re.finditer(r'([A-Za-z]+)(\d)([#b]?)\(([0-9.]+)\)', notes_part):
            raw_pitch = m.group(1).lower()
            pitch = pitch_map.get(raw_pitch)
            if not pitch:
                continue
            note = {
                "pitch": pitch,
                "octave": int(m.group(2)),
                "duration": float(m.group(4)),
                "accidental": m.group(3) if m.group(3) else None,
                "staffIndex": staff_idx,
                "confidence": 0.9
            }
            notes.append(note)
            all_notes.append(note)

        if notes:
            staves.append({"index": staff_idx, "noteCount": len(notes)})

    return all_notes, staves


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

        # Görüntüyü iyileştir
        img = enhance_image(img)

        final_b64 = image_to_b64(img)
        text, err = ask_gpt(final_b64, staff_count)

        if err:
            return jsonify({"success": False, "error": "API hatasi: " + err})
        if not text:
            return jsonify({"success": False, "error": "GPT bos yanit verdi."})

        all_notes, all_staves = parse_notes(text)

        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadi.",
                "debug_raw": text
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
                    "rawText": text
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
