from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
from PIL import Image
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Gorseldeki TUM porteleri AYRI AYRI ve EKSIKSIZ oku.

Her portedeki her notayi soldan saga, ATLAMADAN yaz.

FORMAT: IsimOktav(sure)
ORNEKLER: Do4(1) Re4(0.5) Mib4(0.5) Sol4(1) Fa4#(2) La4(1.5)

SOL ANAHTARI NOTA POZISYONLARI (ezberle):
- Ek cizgi altinda = Do4
- 1. aralik = Re4
- 1. cizgi = Mi4
- 1-2. aralik = Fa4
- 2. cizgi = Sol4
- 2-3. aralik = La4
- 3. cizgi = Si4
- 3-4. aralik = Do5
- 4. cizgi = Re5
- 4-5. aralik = Mi5
- 5. cizgi = Fa5

SURELER:
- Birlik = 4
- Ikilik = 2
- Dortluk = 1
- Sekizlik = 0.5
- Onaltilik = 0.25
- Noktalı dortluk = 1.5
- Noktalı ikilik = 3

AKSIDANLAR:
- Diyez = # (notanin hemen arkasina: Fa4#)
- Bemol = b (notanin hemen arkasina: Sib4)
- Armurdeki diyez/bemoller tum o notaya uygulanir, her seferinde tekrar yazma

KURALLAR:
- Sus isaretlerini ATLA, sadece notalar
- Her porte icin ayri satir yaz
- Sadece asagidaki formatta yaz, baska hicbir sey yazma:

PORTE 1: Do4(1) Re4(0.5) Mi4(0.5) ...
PORTE 2: Sol4(1) La4(1) ...
PORTE 3: ..."""


def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask_gpt(b64):
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
                    {"type": "text", "text": PROMPT}
                ]
            }]
        },
        timeout=120
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result:
        return None, "Bos yanit"
    return result["choices"][0]["message"]["content"].strip(), None


def parse_notes(text, staff_offset=0):
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
        staff_idx = (int(staff_num.group()) - 1 + staff_offset) if staff_num else len(staves)
        notes_part = line[colon + 1:].strip()
        notes = []

        # Format: IsimOktav(sure) veya IsimOktavAksidens(sure)
        # Ornekler: Do4(1)  Fa4#(0.5)  Sib4(1)  Mib5(0.5)
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
        w, h = img.size
        ratio = h / w

        all_notes = []
        all_staves = []
        raw_parts = []

        if ratio > 1.2:
            # Uzun goruntu: uce bol (daha iyi kapsama)
            third = h // 3
            crops = [
                img.crop((0, 0, w, third)),
                img.crop((0, third, w, third * 2)),
                img.crop((0, third * 2, w, h)),
            ]
            for i, crop in enumerate(crops):
                text, err = ask_gpt(image_to_b64(crop))
                if text and not err:
                    raw_parts.append(text)
                    offset = len(all_staves)
                    notes_part, staves_part = parse_notes(text, staff_offset=offset)
                    all_notes.extend(notes_part)
                    all_staves.extend(staves_part)
        else:
            text, err = ask_gpt(image_to_b64(img))
            if err:
                return jsonify({"success": False, "error": "API hatasi: " + err})
            if text:
                raw_parts.append(text)
                all_notes, all_staves = parse_notes(text)

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi. Daha net bir fotograf deneyin."})

        staff_count = len(set(n.get("staffIndex", 0) for n in all_notes))
        raw_text = '\n\n'.join(raw_parts)

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
                    "rawText": raw_text
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    """Şimdilik devre dışı - PDF özelliği sonra eklenecek."""
    return jsonify({"success": False, "error": "PDF özelliği yakında eklenecek."}), 501


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
