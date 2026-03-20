from flask import Flask, request, jsonify
from google import genai
import base64
import json
import re
import os

app = Flask(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """Bu bir nota kagidi gorseli. Gorseldeki TUM notalari sirasiyla oku.

Her porte (satir) icin notalari soldan saga oku.

JSON formatinda dondur, baska hicbir sey yazma:

{
  "notes": [
    {"pitch": "Do", "octave": 4, "duration": 1, "staffIndex": 0},
    {"pitch": "Re", "octave": 4, "duration": 1, "staffIndex": 0}
  ],
  "staffCount": 3,
  "timeSignature": "4/4",
  "clef": "treble"
}

Kurallar:
- pitch: Turkce nota adi (Do, Re, Mi, Fa, Sol, La, Si)
- octave: 3, 4 veya 5
- duration: birlik=4, ikilik=2, dortluk=1, sekizlik=0.5, onaltilik=0.25
- staffIndex: hangi porte satirinda (0 dan basla)
- Diyez varsa pitch e # ekle (orn: Fa#)
- Bemol varsa pitch e b ekle (orn: Sib)
- Sus isaretlerini atla
- staffCount: toplam porte satir sayisi
- HER notayi atlamadan yaz, tum porteleri oku
- Sadece JSON dondur, aciklama yazma"""

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "NotaVision OCR - Gemini"})

@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Goruntu verisi eksik"}), 400

        img_bytes = base64.b64decode(data['image'])

        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=[
                PROMPT,
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}
            ]
        )

        text = response.text.strip()

        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "AI yaniti parse edilemedi"})

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
                    "clef": result.get("clef", "treble"),
                    "engine": "gemini-2.0-flash-lite"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
