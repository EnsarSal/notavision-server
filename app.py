from flask import Flask, request, jsonify
import httpx
import json
import re
import os

app = Flask(__name__)
API_KEY = os.environ.get("GEMINI_API_KEY")

PROMPT = """Bu bir nota kagidi gorseli. Lutfen asagidaki kurallara gore TUM notalari oku:

GOREV:
- Gorseldeki her porte satirini AYRI AYRI oku (kac porte varsa o kadar grup olustur)
- Her portedeki notalari SOLDAN SAGA, DOGRU SIRAYA gore yaz
- Her notanin PITCH (isim), OKTAV, SURE ve hangi PORTE oldugunu belirt
- Notanin portedeki konumunu (cizgi/aralik) da belirt

NOTA TANIMLAMA:
- Sol anahtari kullaniliyor
- Cizgiler asagidan yukari: Mi4, Sol4, Si4, Re5, Fa5
- Aralidar asagidan yukari: Fa3(alt), La3(alt), Do4, Mi4(aralik), Sol4(aralik), Si4(aralik), Re5(aralik)
- Porte altindaki ek cizgiler: Do4 (1.ek cizgi), La3, Sol3 vs
- Porte ustundeki ek cizgiler: La5, Si5 vs
- Diyez (#) ve bemol (b) isaretlerine dikkat et
- Armure (bas taraftaki diyez/bemoller) tum notalar icin gecerli

JSON formatinda dondur:
{
  "title": "eser adi varsa",
  "clef": "treble",
  "timeSignature": "4/4",
  "keySignature": "1#",
  "staffCount": 4,
  "staves": [
    {
      "staffIndex": 0,
      "notes": [
        {
          "pitch": "Mi",
          "octave": 4,
          "duration": 1,
          "durationName": "dortluk",
          "position": "1.cizgi",
          "accidental": null
        }
      ]
    }
  ]
}

KURALLAR:
- pitch: Turkce nota adi (Do, Re, Mi, Fa, Sol, La, Si)
- octave: 3, 4 veya 5
- duration: birlik=4, ikilik=2, dortluk=1, sekizlik=0.5, onaltilik=0.25
- durationName: birlik/ikilik/dortluk/sekizlik/onaltilik
- position: notanin portedeki yeri (1.cizgi, 2.aralik, ust ek cizgi vs)
- accidental: null, "#" veya "b"
- Sus isaretlerini atla
- Bagli notalari (tie/slur) ayri ayri yaz
- Her portey ayri staffIndex ile ver (0,1,2,3...)
- Maksimum 120 nota
- SADECE JSON dondur, baska hicbir sey yazma"""

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
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64
                            }
                        }
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
            return jsonify({"success": False, "error": "API bos yanit verdi: " + json.dumps(result)[:500]})

        finish_reason = candidates[0].get("finishReason", "")
        text = candidates[0]["content"]["parts"][0]["text"].strip()

        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # JSON kesilmisse tamamla
        if finish_reason == "MAX_TOKENS":
            last_brace = text.rfind('},')
            if last_brace > 0:
                text = text[:last_brace+1] + ']}]}'

        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return jsonify({"success": False, "error": "Yanit parse edilemedi: " + text[:300]})

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raw = json_match.group()
            last_valid = raw.rfind('},')
            if last_valid > 0:
                fixed = raw[:last_valid+1] + ']}]}'
                try:
                    parsed = json.loads(fixed)
                except:
                    return jsonify({"success": False, "error": "JSON parse hatasi: " + str(e)})
            else:
                return jsonify({"success": False, "error": "JSON parse hatasi: " + str(e)})

        staves = parsed.get("staves", [])
        if not staves:
            return jsonify({"success": False, "error": "Porte bulunamadi"})

        # Tum notaları düz listeye de cevir (geriye donuk uyumluluk)
        all_notes = []
        for stave in staves:
            for note in stave.get("notes", []):
                note["staffIndex"] = stave["staffIndex"]
                note["confidence"] = 0.9
                all_notes.append(note)

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": [{"index": s["staffIndex"], "spacing": 12, "notes": s.get("notes", [])} for s in staves],
                "metadata": {
                    "title": parsed.get("title", ""),
                    "noteCount": len(all_notes),
                    "staffCount": parsed.get("staffCount", len(staves)),
                    "timeSignature": parsed.get("timeSignature", "4/4"),
                    "keySignature": parsed.get("keySignature", ""),
                    "clef": parsed.get("clef", "treble")
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
