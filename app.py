from flask import Flask, request, jsonify
import google.generativeai as genai
import base64
import json
import re

app = Flask(__name__)

import os
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """Bu bir nota kağıdı görseli. Görseldeki TÜM notaları sırasıyla oku.

Her porte (satır) için notaları soldan sağa oku.

JSON formatında döndür, başka hiçbir şey yazma:

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
- pitch: Türkçe nota adı (Do, Re, Mi, Fa, Sol, La, Si)
- octave: 3, 4 veya 5
- duration: birlik=4, ikilik=2, dörtlük=1, sekizlik=0.5, onaltılık=0.25
- staffIndex: hangi porte satırında (0'dan başla)
- Diyez varsa pitch'e # ekle (örn: "Fa#")
- Bemol varsa pitch'e b ekle (örn: "Si b")
- Sus işaretlerini atla
- staffCount: toplam porte satır sayısı
- HER notayı atlamadan yaz, tüm porteleri oku
- Sadece JSON döndür, açıklama yazma"""

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "NotaVision OCR - Gemini"})

@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Görüntü verisi eksik"}), 400

        img_bytes = base64.b64decode(data['image'])

        model = genai.GenerativeModel('gemini-1.5-flash')

        response = model.generate_content([
            PROMPT,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])

        text = response.text.strip()
        
        # JSON'u parse et
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "AI yanıtı parse edilemedi"})

        result = json.loads(json_match.group())
        notes = result.get("notes", [])
        
        if not notes:
            return jsonify({"success": False, "error": "Nota bulunamadı"})

        # Confidence ekle
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
                    "engine": "gemini-2.0-flash"
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
