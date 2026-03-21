from flask import Flask, request, jsonify
import httpx
import json
import re
import os

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

PROMPT = """Nota kagidi gorseli. TUM porte satirlarini oku. Kompakt JSON dondur.

SOL ANAHTARI POZISYONLARI:
Do4=ek cizgi, Re4=1.aralik, Mi4=1.cizgi, Fa4=1-2aralik, Sol4=2.cizgi, La4=2-3aralik, Si4=3.cizgi, Do5=3-4aralik, Re5=4.cizgi, Mi5=4-5aralik, Fa5=5.cizgi

SURE: 0.25=onaltilik, 0.5=sekizlik, 1=dortluk, 2=ikilik, 4=birlik

ARMURE: Bastaki diyez/bemoller tum ilgili notalara uygulanir.

SADECE bu JSON formatini dondur (baska hicbir sey yazma, markdown kullanma):
{"title":"","time":"4/4","key":"1#","staves":[{"s":0,"n":[{"p":"Mi","o":4,"d":1,"a":null}]},{"s":1,"n":[{"p":"Do","o":5,"d":0.5,"a":null}]}]}

KURALLAR:
- p: Do/Re/Mi/Fa/Sol/La/Si
- o: 3/4/5
- d: 0.25/0.5/1/1.5/2/3/4
- a: null veya "#" veya "b"
- Sus isaretlerini atla
- Her porte ayri s degeri (0dan basla)
- SADECE JSON, markdown yok"""

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

        finish_reason = candidates[0].get("finishReason", "")
        text = candidates[0]["content"]["parts"][0]["text"].strip()

        # Markdown temizle
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Kesik JSON tamir et
        if finish_reason == "MAX_TOKENS" or not text.rstrip().endswith('}'):
            last = text.rfind(',"a"')
            if last > 0:
                end = text.find('}', last)
                if end > 0:
                    text = text[:end+1] + ']}]}'

        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "JSON bulunamadi: " + text[:200]})

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            raw = json_match.group()
            last = raw.rfind(',"a"')
            if last > 0:
                end = raw.find('}', last)
                if end > 0:
                    try:
                        parsed = json.loads(raw[:end+1] + ']}]}')
                    except:
                        return jsonify({"success": False, "error": "JSON tamir edilemedi"})
            else:
                return jsonify({"success": False, "error": "JSON parse hatasi"})

        staves = parsed.get("staves", [])
        if not staves:
            return jsonify({"success": False, "error": "Porte bulunamadi"})

        all_notes = []
        for stave in staves:
            for note in stave.get("n", []):
                all_notes.append({
                    "pitch": note.get("p", "Do"),
                    "octave": note.get("o", 4),
                    "duration": note.get("d", 1),
                    "accidental": note.get("a"),
                    "staffIndex": stave["s"],
                    "confidence": 0.9
                })

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": [{"index": s["s"], "noteCount": len(s.get("n", []))} for s in staves],
                "metadata": {
                    "title": parsed.get("title", ""),
                    "noteCount": len(all_notes),
                    "staffCount": len(staves),
                    "timeSignature": parsed.get("time", "4/4"),
                    "keySignature": parsed.get("key", ""),
                    "clef": "treble"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
