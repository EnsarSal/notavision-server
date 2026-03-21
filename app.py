from flask import Flask, request, jsonify
import httpx
import base64
import json
import re
import os
from PIL import Image
import io

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Sadece bu portedeki notalari soldan saga oku.
Her notayi su formatta yaz: IsimOktav(sure)
Ornek: Do4(1) Re4(0.5) Mi4b(0.5) Sol4(1) Fa4#(2)

SOL ANAHTARI: Do4=ek cizgi, Re4=1.aralik, Mi4=1.cizgi, Fa4=1-2aralik, Sol4=2.cizgi, La4=2-3aralik, Si4=3.cizgi, Do5=3-4aralik, Re5=4.cizgi, Mi5=4-5aralik
SURELER: 4=birlik, 2=ikilik, 1=dortluk, 0.5=sekizlik, 0.25=onaltilik
ARMURE: Bastaki diyez/bemoller tum ilgili notalara uygulanir.

SADECE nota listesini yaz, baska hicbir sey yazma. Sus isaretlerini atla."""

def crop_staff(img, staff_idx, total_staffs):
    w, h = img.size
    margin = 0.05
    staff_h = (1 - 2 * margin) / total_staffs
    top = int((margin + staff_idx * staff_h) * h)
    bottom = int((margin + (staff_idx + 1) * staff_h) * h)
    return img.crop((0, top, w, bottom))

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def ask_gemini(b64):
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]}],
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.1}
        },
        timeout=60
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    candidates = result.get("candidates", [])
    if not candidates:
        return None, "Bos yanit"
    return candidates[0]["content"]["parts"][0]["text"].strip(), None

def parse_notes(text, staff_idx):
    notes = []
    pitch_map = {'do':'Do','re':'Re','mi':'Mi','fa':'Fa','sol':'Sol','la':'La','si':'Si'}
    for m in re.finditer(r'([A-Za-z]+)(\d)([#b]?)\(([0-9.]+)\)', text):
        pitch = pitch_map.get(m.group(1).lower())
        if not pitch:
            continue
        notes.append({
            "pitch": pitch,
            "octave": int(m.group(2)),
            "duration": float(m.group(4)),
            "accidental": m.group(3) or None,
            "staffIndex": staff_idx,
            "confidence": 0.9
        })
    return notes

def detect_staff_count(img):
    # Goruntunun yuksekligine gore porte sayisini tahmin et
    w, h = img.size
    ratio = h / w
    if ratio < 0.4:
        return 2
    elif ratio < 0.7:
        return 3
    elif ratio < 1.0:
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

        # Gorseli yukle
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Porte sayisini tespit et
        staff_count = data.get('staffCount') or detect_staff_count(img)

        all_notes = []
        staves_info = []

        for i in range(staff_count):
            # Her porteyi ayri kirp
            cropped = crop_staff(img, i, staff_count)
            b64 = image_to_b64(cropped)

            # Gemini'ye sor
            text, err = ask_gemini(b64)
            if err or not text:
                continue

            # Notalari parse et
            notes = parse_notes(text, i)
            if notes:
                all_notes.extend(notes)
                staves_info.append({"index": i, "noteCount": len(notes), "raw": text})

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": staves_info,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": len(staves_info),
                    "timeSignature": "4/4",
                    "clef": "treble"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
