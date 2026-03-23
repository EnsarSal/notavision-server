from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image, ImageEnhance, ImageFilter
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ADIM 1: Sadece metni oku
TEXT_PROMPT = """This is a music sheet image. Below each note there are syllable texts written in small font.

Please read ALL the text written below the notes, line by line from left to right.
Just read the text as-is. Example output:

Line 1: es do re re mi do re do mi re re es do re mi do do si si re si do
Line 2: es si do re si si la do si si es la la do si si la la la
Line 3: es re re sol fa# fa# sol mi mi es mi fa# sol mi mi re re fa mi mi
Line 4: es mi mi fa re re mi do re do mi re re es do re mi do do si si re si do

Read every line of text you see below the notes. Do not skip any."""

# ADIM 2: Metni notaya çevir
def build_parse_prompt(raw_text, staff_count):
    return f"""Convert this solfege text into note format.

Solfege text from the music sheet:
{raw_text}

Rules:
- "es" = Mib4 (duration 0.5)
- "sib" = Sib4
- "fa#" = Fa4#
- "do#" = Do4#
- Each syllable is a quarter note (1) by default unless it looks longer
- Repeated syllables with underscore = longer note
- Use octave 4 for middle range, 5 for high notes (sol, la, si going up)

Convert each line to this exact format:
PORTE 1: Do4(1) Re4(1) Mib4(0.5) ...
PORTE 2: ...

Write exactly {staff_count} PORTE lines. Nothing else."""


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
    w, h = img.size
    if w < 1500:
        scale = 1500 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    if w > 2500 or h > 2500:
        scale = min(2500 / w, 2500 / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = img.convert('L').convert('RGB')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def ask_gpt_text(b64):
    """Adım 1: Görselden solfej metnini oku."""
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "max_tokens": 2048,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": TEXT_PROMPT}
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


def ask_gpt_parse(raw_text, staff_count):
    """Adım 2: Metni PORTE formatına çevir."""
    prompt = build_parse_prompt(raw_text, staff_count)
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60
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
        img = enhance_image(img)
        final_b64 = image_to_b64(img)

        # ADIM 1: Solfej metnini oku
        raw_text, err = ask_gpt_text(final_b64)
        if err:
            return jsonify({"success": False, "error": "API hatasi (step1): " + err})
        if not raw_text:
            return jsonify({"success": False, "error": "Metin okunamadi."})

        # ADIM 2: Metni PORTE formatına çevir
        porte_text, err2 = ask_gpt_parse(raw_text, staff_count)
        if err2:
            return jsonify({"success": False, "error": "API hatasi (step2): " + err2})
        if not porte_text:
            return jsonify({"success": False, "error": "Format donusturme hatasi.", "debug_raw": raw_text})

        all_notes, all_staves = parse_notes(porte_text)

        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadi.",
                "debug_raw": raw_text + "\n\n---\n\n" + porte_text
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
                    "rawText": raw_text
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
