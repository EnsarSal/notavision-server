from flask import Flask, request, jsonify
import httpx
import base64
import json
import re
import os
from PIL import Image
import io

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import black, white, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

# ─── NOTA POZİSYON HARİTASI (sol anahtarı) ───
# Değer: porte ortasından (3. çizgi = Si4) yukarı kaç yarım adım
NOTE_STAFF_POS = {
    'Do3': -7, 'Re3': -6, 'Mi3': -5, 'Fa3': -4, 'Sol3': -3, 'La3': -2, 'Si3': -1,
    'Do4':  0, 'Re4':  1, 'Mi4':  2, 'Fa4':  3, 'Sol4':  4, 'La4':  5, 'Si4':  6,
    'Do5':  7, 'Re5':  8, 'Mi5':  9, 'Fa5': 10, 'Sol5': 11, 'La5': 12, 'Si5': 13,
}

def image_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def ask_gpt(b64):
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
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                    },
                    {"type": "text", "text": PROMPT}
                ]
            }]
        },
        timeout=120
    )
    result = resp.json()
    if "error" in result:
        return None, str(result["error"])
    if "choices" not in result:
        return None, "Bos yanit"
    return result["choices"][0]["message"]["content"].strip(), None

def parse_notes(text, staff_offset=0):
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
        staff_idx = (int(staff_num.group()) - 1 + staff_offset) if staff_num else len(staves)
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

# ═══════════════════════════════════════════════
#  PDF ÜRETİCİ
# ═══════════════════════════════════════════════

def duration_to_name(dur):
    """Süre değerinden nota adı döner."""
    mapping = {4: 'whole', 2: 'half', 1: 'quarter', 0.5: 'eighth',
               0.25: 'sixteenth', 1.5: 'dotted_quarter', 3: 'dotted_half'}
    return mapping.get(float(dur), 'quarter')

def note_y(pitch, octave, staff_top, line_spacing):
    """Notanın Y koordinatını hesapla (ReportLab koordinatı — aşağıdan yukarı)."""
    key = f"{pitch}{octave}"
    pos = NOTE_STAFF_POS.get(key, 0)
    # 3. çizgi (Si4, pos=6) = staff_top + 2*line_spacing
    # Her yarım adım = line_spacing/2
    # pos=0 (Do4) → 3. çizginin 3 adım altı
    third_line_y = staff_top + 2 * line_spacing
    return third_line_y + pos * (line_spacing / 2)

def draw_treble_clef(c, x, staff_top, line_spacing):
    """Basit sol anahtarı — büyük G harfi gibi."""
    c.setFont("Helvetica-Bold", line_spacing * 5.5)
    c.drawString(x, staff_top - line_spacing * 0.5, "𝄞")

def draw_staff(c, x_start, x_end, staff_top, line_spacing):
    """5 porte çizgisi çiz."""
    c.setLineWidth(0.8)
    c.setStrokeColor(black)
    for i in range(5):
        y = staff_top + i * line_spacing
        c.line(x_start, y, x_end, y)

def draw_note_head(c, x, y, filled=True, line_spacing=8):
    """Nota başı çiz (elips)."""
    rx = line_spacing * 0.52
    ry = line_spacing * 0.38
    c.setFillColor(black)
    c.setStrokeColor(black)
    c.setLineWidth(1.2)
    if filled:
        c.ellipse(x - rx, y - ry, x + rx, y + ry, fill=1, stroke=0)
    else:
        c.ellipse(x - rx, y - ry, x + rx, y + ry, fill=0, stroke=1)

def draw_stem(c, x, y, up=True, line_spacing=8):
    """Nota sapı çiz."""
    stem_len = line_spacing * 3.5
    c.setLineWidth(1.2)
    c.setStrokeColor(black)
    if up:
        c.line(x + line_spacing * 0.48, y, x + line_spacing * 0.48, y + stem_len)
    else:
        c.line(x - line_spacing * 0.48, y, x - line_spacing * 0.48, y - stem_len)

def draw_flag(c, x, y, up=True, count=1, line_spacing=8):
    """Sekizlik/onaltılık bayrak."""
    stem_len = line_spacing * 3.5
    sx = x + (line_spacing * 0.48 if up else -line_spacing * 0.48)
    sy = y + stem_len if up else y - stem_len
    c.setLineWidth(1.2)
    c.setStrokeColor(black)
    for i in range(count):
        fy = sy - i * line_spacing * 0.7 if up else sy + i * line_spacing * 0.7
        c.bezier(sx, fy,
                 sx + line_spacing * 0.8, fy + (line_spacing * 0.4 if up else -line_spacing * 0.4),
                 sx + line_spacing * 0.6, fy + (line_spacing * 0.9 if up else -line_spacing * 0.9),
                 sx + line_spacing * 0.2, fy + (line_spacing * 1.2 if up else -line_spacing * 1.2))

def draw_ledger_lines(c, x, note_y_pos, staff_top, staff_bottom, line_spacing):
    """Ek çizgiler (Do4 gibi çizgi dışı notalar için)."""
    lw = line_spacing * 1.4
    c.setLineWidth(0.8)
    c.setStrokeColor(black)
    # Altındaki ek çizgiler
    y = staff_top - line_spacing
    while y >= note_y_pos - line_spacing * 0.3:
        c.line(x - lw / 2, y, x + lw / 2, y)
        y -= line_spacing
    # Üstündeki ek çizgiler
    y = staff_bottom + line_spacing
    while y <= note_y_pos + line_spacing * 0.3:
        c.line(x - lw / 2, y, x + lw / 2, y)
        y += line_spacing

def draw_accidental(c, x, y, acc, line_spacing):
    """Diyez veya bemol."""
    c.setFont("Helvetica-Bold", line_spacing * 1.4)
    c.setFillColor(black)
    symbol = "♯" if acc == "#" else "♭"
    c.drawString(x - line_spacing * 1.6, y - line_spacing * 0.4, symbol)

def draw_note(c, note, x, staff_top, line_spacing):
    """Tek bir notayı çiz."""
    pitch = note.get("pitch", "Do")
    octave = note.get("octave", 4)
    duration = float(note.get("duration", 1))
    accidental = note.get("accidental")
    dur_name = duration_to_name(duration)

    y = note_y(pitch, octave, staff_top, line_spacing)
    staff_bottom = staff_top + 4 * line_spacing

    # Ek çizgi gerekiyor mu?
    if y < staff_top - line_spacing * 0.3 or y > staff_bottom + line_spacing * 0.3:
        draw_ledger_lines(c, x, y, staff_top, staff_bottom, line_spacing)

    # Aksidens
    if accidental:
        draw_accidental(c, x, y, accidental, line_spacing)

    # Nota başı
    filled = dur_name not in ('whole', 'half', 'dotted_half')
    draw_note_head(c, x, y, filled=filled, line_spacing=line_spacing)

    # Sap ve bayrak
    has_stem = dur_name != 'whole'
    if has_stem:
        up = y < staff_top + 2 * line_spacing  # orta çizginin altındaysa sap yukarı
        draw_stem(c, x, y, up=up, line_spacing=line_spacing)
        if dur_name == 'eighth':
            draw_flag(c, x, y, up=up, count=1, line_spacing=line_spacing)
        elif dur_name == 'sixteenth':
            draw_flag(c, x, y, up=up, count=2, line_spacing=line_spacing)

    # Nokta (noktalı notalar)
    if 'dotted' in dur_name:
        c.setFillColor(black)
        c.circle(x + line_spacing * 0.75, y + line_spacing * 0.2,
                 line_spacing * 0.15, fill=1, stroke=0)

    # Nota adı etiketi (küçük)
    c.setFont("Helvetica", line_spacing * 0.85)
    c.setFillColor(HexColor("#555555"))
    label = f"{pitch}{octave}"
    if accidental:
        label = f"{pitch}{'#' if accidental == '#' else 'b'}{octave}"
    c.drawCentredString(x, y - line_spacing * 1.6, label)

def generate_pdf(notes_data, title="NotaVision", time_signature="4/4"):
    """
    notes_data: [{"pitch":"Do","octave":4,"duration":1,"accidental":null,"staffIndex":0}, ...]
    Döner: bytes (PDF)
    """
    buf = io.BytesIO()
    page_w, page_h = A4  # 595 x 842 pt
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # ── Sayfa ayarları ──
    margin_left = 30 * mm
    margin_right = 20 * mm
    margin_top = 20 * mm
    margin_bottom = 20 * mm

    line_spacing = 8      # porte çizgileri arası (pt)
    staff_height = 4 * line_spacing
    staff_gap = 22 * mm   # porteler arası boşluk
    notes_per_row = 12    # bir porte satırına max nota
    clef_width = 20 * mm
    note_gap = (page_w - margin_left - margin_right - clef_width) / notes_per_row

    # ── Başlık ──
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(black)
    c.drawCentredString(page_w / 2, page_h - margin_top - 10, title)

    # ── Notaları portelere göre grupla ──
    from collections import defaultdict
    staff_notes = defaultdict(list)
    for n in notes_data:
        staff_notes[n.get("staffIndex", 0)].append(n)

    # ── Her porteyi çiz ──
    current_y = page_h - margin_top - 35  # başlangıç Y

    for staff_idx in sorted(staff_notes.keys()):
        notes = staff_notes[staff_idx]

        # Notaları satırlara böl
        rows = [notes[i:i+notes_per_row] for i in range(0, len(notes), notes_per_row)]

        for row_idx, row_notes in enumerate(rows):
            staff_top = current_y - staff_height

            # Yeni sayfa gerekiyor mu?
            if staff_top < margin_bottom + staff_height + 20:
                c.showPage()
                current_y = page_h - margin_top - 10
                staff_top = current_y - staff_height

            x_start = margin_left
            x_end = page_w - margin_right

            # Porte çizgileri
            draw_staff(c, x_start, x_end, staff_top, line_spacing)

            # Sol anahtarı
            c.setFillColor(black)
            c.setFont("Helvetica-Bold", line_spacing * 5)
            c.drawString(x_start + 2, staff_top - line_spacing * 0.2, "𝄞")

            # Zaman işareti (sadece ilk satır, ilk porte)
            if staff_idx == 0 and row_idx == 0:
                ts_parts = time_signature.split('/')
                c.setFont("Helvetica-Bold", line_spacing * 1.6)
                c.setFillColor(black)
                c.drawCentredString(x_start + clef_width + 6,
                                    staff_top + 3 * line_spacing - 2, ts_parts[0])
                c.drawCentredString(x_start + clef_width + 6,
                                    staff_top + line_spacing - 2, ts_parts[1])
                note_x_offset = clef_width + 14 * mm
            else:
                note_x_offset = clef_width + 6 * mm

            # Notaları çiz
            for i, note in enumerate(row_notes):
                nx = x_start + note_x_offset + i * note_gap + note_gap / 2
                draw_note(c, note, nx, staff_top, line_spacing)

            # Son çift çizgi (son satır)
            if row_idx == len(rows) - 1:
                end_x = x_start + note_x_offset + len(row_notes) * note_gap + 10
                c.setLineWidth(1)
                c.line(end_x, staff_top, end_x, staff_top + 4 * line_spacing)
                c.setLineWidth(2.5)
                c.line(end_x + 4, staff_top, end_x + 4, staff_top + 4 * line_spacing)

            current_y = staff_top - staff_gap

    c.save()
    return buf.getvalue()

# ═══════════════════════════════════════════════
#  ENDPOINTLER
# ═══════════════════════════════════════════════

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

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        ratio = h / w

        all_notes = []
        all_staves = []

        if ratio > 1.0:
            top_img = img.crop((0, 0, w, h // 2))
            bot_img = img.crop((0, h // 2, w, h))

            top_b64 = image_to_b64(top_img)
            text1, err1 = ask_gpt(top_b64)
            if text1 and not err1:
                notes1, staves1 = parse_notes(text1, staff_offset=0)
                all_notes.extend(notes1)
                all_staves.extend(staves1)

            bot_b64 = image_to_b64(bot_img)
            text2, err2 = ask_gpt(bot_b64)
            if text2 and not err2:
                offset = len(all_staves)
                notes2, staves2 = parse_notes(text2, staff_offset=offset)
                all_notes.extend(notes2)
                all_staves.extend(staves2)
        else:
            text, err = ask_gpt(b64)
            if err:
                return jsonify({"success": False, "error": "API hatasi: " + err})
            all_notes, all_staves = parse_notes(text)

        if not all_notes:
            return jsonify({"success": False, "error": "Nota bulunamadi"})

        # Kaç porte var
        staff_count = len(set(n.get("staffIndex", 0) for n in all_notes))

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": all_staves,
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": staff_count,
                    "timeSignature": "4/4",
                    "clef": "treble",
                    "rawText": ""
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": "Sunucu hatasi: " + str(e)}), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_endpoint():
    """
    Body: {
      "notes": [...],          // nota dizisi
      "title": "Parça Adı",    // opsiyonel
      "timeSignature": "4/4"   // opsiyonel
    }
    Döner: { "success": true, "pdf": "<base64>" }
    """
    try:
        data = request.json
        if not data or 'notes' not in data:
            return jsonify({"success": False, "error": "Notalar eksik"}), 400

        notes = data['notes']
        title = data.get('title', 'NotaVision')
        time_sig = data.get('timeSignature', '4/4')

        if not notes:
            return jsonify({"success": False, "error": "Nota listesi bos"}), 400

        pdf_bytes = generate_pdf(notes, title=title, time_signature=time_sig)
        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')

        return jsonify({
            "success": True,
            "pdf": pdf_b64,
            "pageCount": 1
        })

    except Exception as e:
        return jsonify({"success": False, "error": "PDF hatasi: " + str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
