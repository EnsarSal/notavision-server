from flask import Flask, request, jsonify
import httpx
import base64
import re
import os
import subprocess
import tempfile
from PIL import Image
import io
from collections import defaultdict

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MUSICXML_PROMPT = """You are a professional Optical Music Recognition (OMR) system.

Analyze this sheet music image carefully and convert it to valid MusicXML format.

Rules:
- Detect ALL staves and ALL notes without skipping any
- Detect clef (treble/bass), time signature, key signature (sharps/flats)
- For each note detect: pitch (C,D,E,F,G,A,B), octave (1-7), duration (whole/half/quarter/eighth/sixteenth), accidentals (#/b/natural)
- Detect dotted notes
- Group notes into measures using barlines
- Use standard MusicXML 4.0 partwise format

Return ONLY valid MusicXML. No explanation, no markdown, no code blocks. Start directly with <?xml"""

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def ask_gpt_musicxml(b64_images):
    content = []
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
        })
    content.append({"type": "text", "text": MUSICXML_PROMPT})

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": "gpt-4o", "max_tokens": 8192, "messages": [{"role": "user", "content": content}]},
        timeout=180
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result:
        return None, "Boş yanıt"
    return result["choices"][0]["message"]["content"].strip(), None

def clean_musicxml(raw):
    raw = re.sub(r'```xml\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()
    idx = raw.find('<?xml')
    if idx > 0:
        raw = raw[idx:]
    return raw

def musicxml_to_notes_json(musicxml_str):
    notes = []
    pitch_map = {'C':'Do','D':'Re','E':'Mi','F':'Fa','G':'Sol','A':'La','B':'Si'}
    dur_map = {'whole':4,'half':2,'quarter':1,'eighth':0.5,'16th':0.25,'sixteenth':0.25}
    parts = re.findall(r'<part[^>]*>(.*?)</part>', musicxml_str, re.DOTALL)
    for p_idx, part in enumerate(parts):
        note_blocks = re.findall(r'<note>(.*?)</note>', part, re.DOTALL)
        for nb in note_blocks:
            if '<rest' in nb:
                continue
            step_m = re.search(r'<step>([A-G])</step>', nb)
            oct_m = re.search(r'<octave>(\d)</octave>', nb)
            alter_m = re.search(r'<alter>([-\d.]+)</alter>', nb)
            type_m = re.search(r'<type>(.*?)</type>', nb)
            dot_m = re.search(r'<dot/>', nb)
            if not step_m or not oct_m:
                continue
            pitch = pitch_map.get(step_m.group(1), step_m.group(1))
            octave = int(oct_m.group(1))
            alter = float(alter_m.group(1)) if alter_m else 0
            ntype = type_m.group(1).strip() if type_m else 'quarter'
            duration = dur_map.get(ntype, 1)
            if dot_m:
                duration *= 1.5
            accidental = '#' if alter == 1 else ('b' if alter == -1 else None)
            notes.append({
                "pitch": pitch, "octave": octave, "duration": duration,
                "accidental": accidental, "staffIndex": p_idx, "confidence": 0.92
            })
    return notes

def musicxml_to_pdf_via_lilypond(musicxml_str, title="NotaVision"):
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = os.path.join(tmpdir, 'score.xml')
        ly_path = os.path.join(tmpdir, 'score.ly')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(musicxml_str)

        # musicxml2ly ile dönüştür
        mx2ly = subprocess.run(
            ['musicxml2ly', '--output=' + ly_path, xml_path],
            capture_output=True, text=True, timeout=60
        )

        if not os.path.exists(ly_path):
            # Fallback: direkt LilyPond syntax üret
            ly_content = musicxml_to_lilypond_direct(musicxml_str, title)
            with open(ly_path, 'w', encoding='utf-8') as f:
                f.write(ly_content)
        else:
            # Başlık ekle
            with open(ly_path, 'r', encoding='utf-8') as f:
                ly_content = f.read()
            if '\\header' not in ly_content:
                ly_content = f'\\header {{ title = "{title}" tagline = "NotaVision" }}\n' + ly_content
            else:
                ly_content = ly_content.replace('\\header {', f'\\header {{ title = "{title}"')
            with open(ly_path, 'w', encoding='utf-8') as f:
                f.write(ly_content)

        # LilyPond ile PDF üret
        out_base = os.path.join(tmpdir, 'score')
        result = subprocess.run(
            ['lilypond', '--output=' + out_base, ly_path],
            capture_output=True, text=True, timeout=120
        )

        pdf_path = out_base + '.pdf'
        if not os.path.exists(pdf_path):
            raise Exception(f"LilyPond hatası: {result.stderr[:500]}")

        with open(pdf_path, 'rb') as f:
            return f.read()

def musicxml_to_lilypond_direct(musicxml_str, title="NotaVision"):
    step_map = {'C':'c','D':'d','E':'e','F':'f','G':'g','A':'a','B':'b'}
    type_map = {'whole':'1','half':'2','quarter':'4','eighth':'8','16th':'16','sixteenth':'16'}
    oct_map = {1:',,,',2:',,',3:',',4:"'",5:"''",6:"'''",7:"''''"}

    # Zaman işareti
    time_m = re.search(r'<beats>(\d+)</beats>.*?<beat-type>(\d+)</beat-type>', musicxml_str, re.DOTALL)
    time_str = f"{time_m.group(1)}/{time_m.group(2)}" if time_m else "4/4"

    # Armür
    fifths_m = re.search(r'<fifths>([-\d]+)</fifths>', musicxml_str)
    fifths = int(fifths_m.group(1)) if fifths_m else 0
    key_map = {
        0:'c \\major',1:'g \\major',2:'d \\major',3:'a \\major',4:'e \\major',
        5:'b \\major',6:'fis \\major',7:'cis \\major',
        -1:'f \\major',-2:'bes \\major',-3:'ees \\major',
        -4:'aes \\major',-5:'des \\major',-6:'ges \\major',-7:'ces \\major'
    }
    key_str = f"\\key {key_map.get(fifths, 'c \\major')}"

    parts = re.findall(r'<part[^>]*>(.*?)</part>', musicxml_str, re.DOTALL)
    staves = ''

    for part in parts:
        measures = re.findall(r'<measure[^>]*>(.*?)</measure>', part, re.DOTALL)
        ly_notes = []
        for measure in measures:
            note_blocks = re.findall(r'<note>(.*?)</note>', measure, re.DOTALL)
            for nb in note_blocks:
                dot = '.' if re.search(r'<dot/>', nb) else ''
                type_m2 = re.search(r'<type>(.*?)</type>', nb)
                ntype = type_m2.group(1).strip() if type_m2 else 'quarter'
                dur = type_map.get(ntype, '4')

                if '<rest' in nb:
                    ly_notes.append(f"r{dur}{dot}")
                    continue

                step_m = re.search(r'<step>([A-G])</step>', nb)
                oct_m2 = re.search(r'<octave>(\d)</octave>', nb)
                alter_m = re.search(r'<alter>([-\d.]+)</alter>', nb)
                if not step_m or not oct_m2:
                    continue

                step = step_map.get(step_m.group(1), 'c')
                octave = int(oct_m2.group(1))
                alter = float(alter_m.group(1)) if alter_m else 0

                if alter == 1:
                    step += 'is'
                elif alter == -1:
                    step += 'es'

                oct_str = oct_map.get(octave, "'")
                ly_notes.append(f"{step}{oct_str}{dur}{dot}")

            ly_notes.append('|')

        voice = ' '.join(ly_notes)
        staves += f"""
  \\new Staff {{
    {key_str}
    \\time {time_str}
    \\clef treble
    {voice}
  }}
"""

    return f"""\\version "2.24.0"
\\header {{
  title = "{title}"
  tagline = "NotaVision"
}}
\\score {{
  <<
{staves}
  >>
  \\layout {{ }}
}}
"""

def notes_to_musicxml(notes, title="NotaVision"):
    pitch_map = {'Do':'C','Re':'D','Mi':'E','Fa':'F','Sol':'G','La':'A','Si':'B'}
    dur_map = {4:'whole',2:'half',1:'quarter',0.5:'eighth',0.25:'16th',1.5:'quarter',3:'half'}
    div_map = {4:16,2:8,1:4,0.5:2,0.25:1,1.5:6,3:12}

    staff_notes = defaultdict(list)
    for n in notes:
        staff_notes[n.get('staffIndex',0)].append(n)

    parts_xml = ''
    part_list_xml = ''
    for idx, (staff_idx, snotes) in enumerate(sorted(staff_notes.items())):
        pid = f'P{idx+1}'
        part_list_xml += f'<score-part id="{pid}"><part-name>Staff {idx+1}</part-name></score-part>\n'
        measures_xml = ''
        for m_idx, chunk in enumerate([snotes[i:i+4] for i in range(0,len(snotes),4)]):
            notes_xml = ''
            for n in chunk:
                step = pitch_map.get(n.get('pitch','Do'),'C')
                octave = n.get('octave',4)
                dur = float(n.get('duration',1))
                acc = n.get('accidental')
                ntype = dur_map.get(dur,'quarter')
                divisions = div_map.get(dur,4)
                alter_xml = '<alter>1</alter>' if acc=='#' else ('<alter>-1</alter>' if acc=='b' else '')
                dot_xml = '<dot/>' if dur in (1.5,3) else ''
                notes_xml += f"<note><pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch><duration>{divisions}</duration><type>{ntype}</type>{dot_xml}</note>"
            attr_xml = '<attributes><divisions>4</divisions><key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>' if m_idx==0 else ''
            measures_xml += f'<measure number="{m_idx+1}">{attr_xml}{notes_xml}</measure>'
        parts_xml += f'<part id="{pid}">{measures_xml}</part>'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <work><work-title>{title}</work-title></work>
  <part-list>{part_list_xml}</part-list>
  {parts_xml}
</score-partwise>"""

# ═══ ENDPOINTS ═══

@app.route('/health', methods=['GET'])
def health():
    try:
        r = subprocess.run(['lilypond','--version'], capture_output=True, text=True, timeout=5)
        ly = r.stdout.split('\n')[0] if r.returncode==0 else 'not found'
    except:
        ly = 'not found'
    return jsonify({"status":"ok","lilypond":ly})

@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success":False,"error":"Görüntü eksik"}), 400

        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size

        b64_images = [image_to_b64(img.crop((0,0,w,h//2))), image_to_b64(img.crop((0,h//2,w,h)))] if h/w > 1.2 else [image_to_b64(img)]

        musicxml, err = ask_gpt_musicxml(b64_images)
        if err:
            return jsonify({"success":False,"error":"API hatası: "+err})

        musicxml = clean_musicxml(musicxml)
        notes = musicxml_to_notes_json(musicxml)
        if not notes:
            return jsonify({"success":False,"error":"Nota bulunamadı. Daha net fotoğraf deneyin."})

        staff_count = len(set(n.get("staffIndex",0) for n in notes))
        return jsonify({
            "success": True,
            "data": {
                "notes": notes,
                "musicxml": musicxml,
                "metadata": {"noteCount":len(notes),"staffCount":staff_count,"timeSignature":"4/4","clef":"treble","rawText":""}
            }
        })
    except Exception as e:
        return jsonify({"success":False,"error":"Sunucu hatası: "+str(e)}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"success":False,"error":"Veri eksik"}), 400

        title = data.get('title','NotaVision')
        musicxml = data.get('musicxml')

        if not musicxml:
            notes = data.get('notes',[])
            if not notes:
                return jsonify({"success":False,"error":"MusicXML veya nota listesi gerekli"}), 400
            musicxml = notes_to_musicxml(notes, title)

        musicxml = clean_musicxml(musicxml)
        pdf_bytes = musicxml_to_pdf_via_lilypond(musicxml, title)
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
        return jsonify({"success":True,"pdf":pdf_b64})

    except Exception as e:
        return jsonify({"success":False,"error":"PDF hatası: "+str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
