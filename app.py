from flask import Flask, request, jsonify
import httpx
import base64
import json
import re
import os

app = Flask(__name__)

API_KEY = os.environ.get("OPENROUTER_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Gorseldeki TUM notalari sirasiyla oku.
Her porte satiri icin notalari soldan saga oku.
JSON formatinda dondur, baska hicbir sey yazma:
{
  "notes": [
    {"pitch": "Do", "octave": 4, "duration": 1, "staffIndex": 0}
  ],
  "staffCount": 1,
  "timeSignature": "4/4",
  "clef": "treble"
}
Kurallar:
- pitch: Turkce nota adi (Do, Re, Mi, Fa, Sol, La, Si)
- octave: 3, 4 veya 5
- duration: birlik=4, ikilik=2, dortluk=1, sekizlik=0.5, onaltilik=0.25
- staffIndex: hangi porte satirinda (0 dan basla)
- Diyez varsa # ekle (Fa#), bemol varsa b ekle (Sib)
- Sus isaretlerini atla
- HER notayi atlamadan yaz
- Sadece JSON dondur"""

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
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                "max_tokens": 4000
            },
            timeout=120
        )

        result = resp.json()

        if "error" in result:
            return jsonify({"success": False, "error": "API hatasi: " + str(result["error"])})

        if "choices" not in result or len(result["choices"]) == 0:
            return jsonify({"success": False, "error": "API bos yanit verdi: " + json.dumps(result)[:500]})

        text = result["choices"][0]["message"]["content"].strip()

        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "Yanit parse edilemedi: " + text[:300]})

        parsed = json.loads(json_match.group())
        notes = parsed.get("notes", [])
        if not notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        for n in notes:
            n["confidence"] = 0.9

        return jsonify({
            "success": True,
            "data": {
                "notes": notes,
                "staves": [{"index": i, "spacing": 12} for i in range(parsed.get("staffCount", 1))],
                "metadata": {
                    "noteCount": len(notes),
                    "staffCount": parsed.get("staffCount", 1),
                    "timeSignature": parsed.get("timeSignature", "4/4"),
                    "clef": parsed.get("clef", "treble")
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
