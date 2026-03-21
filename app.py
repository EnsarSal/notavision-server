from flask import Flask, request, jsonify
import httpx
import json
import re
import os

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

def parse_notes(text):
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
        staff_idx = int(staff_num.group()) - 1 if staff_num else len(staves)

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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        }
                    ]
                }]
            },
            timeout=120
        )

        result = resp.json()

        if "error" in result:
            return jsonify({"success": False, "error": "API hatasi: " + str(result["error"])})

        if "choices" not in result:
            return jsonify({"success": False, "error": "Bos yanit: " + json.dumps(result)[:300]})

        text = result["choices"][0]["message"]["content"].strip()

        all_notes, staves = parse_notes(text)

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi. Ham yanit: " + text[:300]})

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "rawText": text,
                "staves": staves,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": len(staves),
                    "timeSignature": "4/4",
                    "clef": "treble"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
