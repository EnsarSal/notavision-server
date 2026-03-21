from flask import Flask, request, jsonify
import httpx
import json
import re
import os

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Asagidaki formatta yaz.

SOL ANAHTARI POZISYONLARI:
Do4=ek cizgi alti, Re4=1.aralik, Mi4=1.cizgi, Fa4=1-2aralik, Sol4=2.cizgi, La4=2-3aralik, Si4=3.cizgi, Do5=3-4aralik, Re5=4.cizgi, Mi5=4-5aralik, Fa5=5.cizgi
Ek cizgiler: Do3, Re3, Mi3 (porte alti), Sol5, La5 (porte ustu)

SURE KODLARI: 4=birlik, 2=ikilik, 1=dortluk, 0.5=sekizlik, 0.25=onaltilik, 1.5=noktalidortluk, 3=noktaliikilik

ARMURE: Bastaki diyez/bemoller tum ilgili notalara uygulanir.

CIKTI FORMATI - Her porte icin bir satir:
PORTE 1: Re4(1) Do4(0.5) Mi4b(0.5) Sol4(1) Fa4#(2)
PORTE 2: Si4(1) Do5(0.5) Re5(1) Mi5(0.5)
...

KURALLAR:
- Her nota: IsimOktav(sure) - ornek: Do4(1) Re5(0.5) Fa4#(1) Mib4(0.5)
- Diyez: # notanin hemen arkasina - Fa4#
- Bemol: b notanin hemen arkasina - Mib4
- Sus isaretlerini ATLA
- Bagli notalari (tie) AYRI AYRI yaz
- Tum porteleri yaz, hic atlama
- Sadece PORTE satirlarini yaz, baska hicbir sey yazma"""

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

def parse_note_text(text):
    staves = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line.startswith('PORTE'):
            continue
        
        # PORTE 1: ... kismini ayir
        colon = line.find(':')
        if colon < 0:
            continue
        
        staff_part = line[:colon]
        notes_part = line[colon+1:].strip()
        
        # Porte indexi al
        staff_num = re.search(r'\d+', staff_part)
        staff_idx = int(staff_num.group()) - 1 if staff_num else len(staves)
        
        # Notalari parse et: Do4(1) Re4#(0.5) Mib5(2) ...
        note_pattern = re.finditer(r'([A-Za-z]+)(\d+)([#b]?)\(([0-9.]+)\)', notes_part)
        
        notes = []
        pitch_map = {
            'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa',
            'sol': 'Sol', 'la': 'La', 'si': 'Si'
        }
        
        for m in note_pattern:
            pitch_raw = m.group(1).lower()
            octave = int(m.group(2))
            accidental = m.group(3) if m.group(3) else None
            duration = float(m.group(4))
            
            pitch = pitch_map.get(pitch_raw)
            if not pitch:
                continue
            
            notes.append({
                "pitch": pitch,
                "octave": octave,
                "duration": duration,
                "accidental": accidental if accidental else None,
                "staffIndex": staff_idx,
                "confidence": 0.9
            })
        
        if notes:
            staves.append({"staffIndex": staff_idx, "notes": notes})
    
    return staves

@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Goruntu eksik"}), 400

        b64 = data['image']

        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": PROMPT},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": 8192,
                    "temperature": 0.1
                }
            },
            timeout=120
        )

        result = resp.json()

        if "error" in result:
            return jsonify({"success": False, "error": "API hatasi: " + str(result["error"])})

        candidates = result.get("candidates", [])
        if not candidates:
            return jsonify({"success": False, "error": "API bos yanit: " + json.dumps(result)[:300]})

        text = candidates[0]["content"]["parts"][0]["text"].strip()

        staves = parse_note_text(text)

        if not staves:
            return jsonify({"success": False, "error": "Nota bulunamadi. Ham yanit: " + text[:300]})

        all_notes = []
        for stave in staves:
            all_notes.extend(stave["notes"])

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "rawText": text,
                "staves": [{"index": s["staffIndex"], "noteCount": len(s["notes"])} for s in staves],
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
