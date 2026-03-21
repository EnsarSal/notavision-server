from flask import Flask, request, jsonify
import httpx
import json
import re
import os

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Lutfen asagidaki talimatlara harfiyen uy.

GOREV: Gorseldeki TUM porte satirlarini AYRI AYRI oku. Her portedeki her notayi ATLAMADAN, SIRAYA GORE yaz.

SOL ANAHTARI NOTA POZISYONLARI (asagidan yukari):
- Porte alti 2. ek cizgi: Do3
- Porte alti 1. ek cizgi alti: Si3  
- Porte alti 1. ek cizgi: Do4
- 1. aralik (porte alti): Re4
- 1. cizgi: Mi4
- 1-2 aralik: Fa4
- 2. cizgi: Sol4
- 2-3 aralik: La4
- 3. cizgi: Si4
- 3-4 aralik: Do5
- 4. cizgi: Re5
- 4-5 aralik: Mi5
- 5. cizgi: Fa5
- Porte ustu 1. ek cizgi alti: Sol5
- Porte ustu 1. ek cizgi: La5

ARMURE (anahtar isaretleri): Porte basindaki diyez/bemoller TUM ilgili notalara uygulanir.

SURE DEGERLERI:
- Dolu bas + 4 kuyruk = onaltilik (0.25)
- Dolu bas + 2 kuyruk = sekizlik (0.5) 
- Dolu bas + kuyruk = dortluk (1)
- Dolu bas kuyruksuz = ikilik (2)
- Bos bas kuyruksuz = birlik (4)
- Noktalı dortluk = 1.5, Noktalı ikilik = 3

CIKTI FORMATI - Sadece bu JSON'u dondur, baska hicbir sey yazma:
{
  "title": "eser adi",
  "timeSignature": "4/4",
  "keySignature": "1#",
  "staffCount": 4,
  "staves": [
    {
      "staffIndex": 0,
      "notes": [
        {"pitch": "Mi", "octave": 4, "duration": 1, "accidental": null}
      ]
    }
  ]
}

KURALLAR:
- pitch: SADECE Turkce (Do/Re/Mi/Fa/Sol/La/Si)
- octave: 3, 4 veya 5
- duration: 0.25 / 0.5 / 1 / 1.5 / 2 / 3 / 4
- accidental: null, "#" veya "b"
- Sus/es isaretlerini ATLA
- Her porte icin AYRI staffIndex kullan (0'dan basla)
- SADECE JSON dondur"""

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

        # DEBUG: Ham yaniti dondur
        return jsonify({
            "success": False,
            "error": "DEBUG finish:" + finish_reason + " | " + text[:800]
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
