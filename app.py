from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DETECT_PROMPT = """This image has rows of horizontal lines with text labels below them.

Count the rows and estimate where each row starts and ends as a percentage of image height (0% = top, 100% = bottom). Include some padding around each row.

Reply ONLY in this exact format, nothing else:
COUNT: 4
ROW1: 5 28
ROW2: 28 52
ROW3: 52 76
ROW4: 76 100"""

def read_prompt(n):
    return f"""This image shows ONE row of horizontal lines with small text labels written below.

Read ALL text labels below the lines from left to right. Do not skip any.

Reply ONLY in this exact format, one line only:
PORTE {n}: Do4(1) Re4(0.5) Mib4(0.5) ...

Rules:
- "es" → Mib4(0.5)
- "sib" → Sib4(1)
- "fa#" or "faş" → Fa4#(1)
- "re#" or "reş" → Re4#(1)
- "sol#" → Sol4#(1)
- Others: capitalize first letter, add octave 4
- Eighth note = 0.5, quarter = 1, half = 2, whole = 4, dotted quarter = 1.5"""


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def ask_gpt(b64, prompt):
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


def crop_staff(img, top_pct, bottom_pct):
    w, h = img.size
    top = max(0, int((top_pct / 100) * h))
    bottom = min(h, int((bottom_pct / 100) * h))
    return img.crop((0, top, w, bottom))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Goruntu eksik"}), 400

        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Çok büyükse küçült
        w, h = img.size
        if w > 2000 or h > 3000:
            scale = min(2000 / w, 3000 / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        full_b64 = image_to_b64(img)

        # ADIM 1: Porte konumlarını tespit et
        detect_text, err = ask_gpt(full_b64, DETECT_PROMPT)
        if err or not detect_text:
            return jsonify({"success": False, "error": "Tespit hatası: " + (err or "boş yanıt")})

        count_match = re.search(r'COUNT:\s*(\d+)', detect_text)
        count = int(count_match.group(1)) if count_match else 4

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

        # ADIM 2: Her porteyi crop edip ayrı GPT'ye gönder
        all_notes = []
        all_staves = []
        raw_parts = []

        for i, row in enumerate(rows):
            cropped = crop_staff(img, row['top'], row['bottom'])
            cropped_b64 = image_to_b64(cropped)

            row_text, err = ask_gpt(cropped_b64, read_prompt(i + 1))
            if err or not row_text:
                continue

            raw_parts.append(row_text)
            notes = parse_notes(row_text, i)
            if notes:
                all_notes.extend(notes)
                all_staves.append({"index": i, "noteCount": len(notes)})

        if not all_notes:
            return jsonify({
                "success": False,
                "error": "Nota bulunamadi.",
                "debug": detect_text + "\n\n" + "\n".join(raw_parts)
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
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    return jsonify({"success": False, "error": "PDF ozelligi yakinda eklenecek."}), 501


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
