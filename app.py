from flask import Flask, request, jsonify
import httpx
import base64
import json
import re
import os
from PIL import Image
import io

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Gorseldeki TUM porteleri AYRI AYRI oku. Her portedeki her notayi ATLAMADAN soldan saga yaz.

Her notayi su formatta yaz: IsimOktav(sure)
Ornek: Do4(1) Re4(0.5) Mib4(0.5) Sol4(1) Fa4#(2)

SOL ANAHTARI POZISYONLARI:
Do4=ek cizgi, Re4=1.aralik, Mi4=1.cizgi, Fa4=1-2aralik, Sol4=2.cizgi, La4=2-3aralik, Si4=3.cizgi, Do5=3-4aralik, Re5=4.cizgi, Mi5=4-5aralik, Fa5=5.cizgi

SURELER: 4=birlik, 2=ikilik, 1=dortluk, 0.5=sekizlik, 0.25=onaltilik, 1.5=noktalidortluk

ARMURE: Bastaki diyez/bemoller tum ilgili notalara uygulanir.

Her porte icin ayri satir yaz:
PORTE 1: Do4(1) Re4(0.5) Mi4(0.5) ...
PORTE 2: Sol4(1) La4(1) ...

Sus isaretlerini atla. Sadece PORTE satirlarini yaz, baska hicbir sey yazma."""

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
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
    pitch_map = {'do':'Do','re':'Re','mi':'Mi','fa':'Fa','sol':'Sol','la':'La','si':'Si'}

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line.upper().startswith('PORTE'):
            continue
        colon = line.find(':')
        if colon < 0:
            continue
        staff_num = re.search(r'\d+', line[:colon])
        staff_idx = (int(staff_num.group()) - 1 + staff_offset) if staff_num else len(staves)
        notes_part = line[colon+1:].strip()
        notes = []
        for m in re.finditer(r'([A-Za-z]+)(\d)([#b]?)\(([0-9.]+)\)', notes_part):
            pitch = pitch_map.get(m.group(1).lower())
            if not pitch:
                continue
            note = {
                "pitch": pitch,
                "octave": int(m.group(2)),
                "duration": float(m.group(4)),
                "accidental": m.group(3) or None,
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

        # Gorseli yukle ve boyutuna bak
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        ratio = h / w

        all_notes = []
        all_staves = []

        if ratio > 1.0:
            # Uzun goruntu - ikiye bol
            top_img = img.crop((0, 0, w, h // 2))
            bot_img = img.crop((0, h // 2, w, h))

            top_b64 = image_to_b64(top_img)
            text1, err1 = ask_gpt(top_b64)
            if text1 and not err1:
                notes1, staves1 = parse_notes(text1, staff_offset=0)
                all_notes.extend(notes1)
                all_staves.extend(staves1)

            bot_b64 = image_to_b64(bot_img)
            text2, err2 = ask_gpt(bot_b64)
            if text2 and not err2:
                offset = len(all_staves)
                notes2, staves2 = parse_notes(text2, staff_offset=offset)
                all_notes.extend(notes2)
                all_staves.extend(staves2)
        else:
            # Normal goruntu - tek istek
            text, err = ask_gpt(b64)
            if err:
                return jsonify({"success": False, "error": "API hatasi: " + err})
            all_notes, all_staves = parse_notes(text)

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": all_staves,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": len(all_staves),
                    "timeSignature": "4/4",
                    "clef": "treble"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```

`requirements.txt`'e de Pillow ekle:
```
flask==3.0.0
httpx==0.27.0
gunicorn==21.2.0
Pillow==11.1.0
