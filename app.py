from flask import Flask, request, jsonify
from openai import OpenAI
import base64
import json
import re
import os

app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("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")
)

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

        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            max_tokens=4000
        )

        text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "Yanit parse edilemedi"})

        result = json.loads(json_match.group())
        notes = result.get("notes", [])
        if not notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        for n in notes:
            n["confidence"] = 0.9

        return jsonify({
            "success": True,
            "data": {
                "notes": notes,
                "staves": [{"index": i, "spacing": 12} for i in range(result.get("staffCount", 1))],
                "metadata": {
                    "noteCount": len(notes),
                    "staffCount": result.get("staffCount", 1),
                    "timeSignature": result.get("timeSignature", "4/4"),
                    "clef": result.get("clef", "treble")
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
