from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "NotaVision OCR"})

@app.route('/process', methods=['POST'])
def process_sheet():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "Görüntü verisi eksik"}), 400

        img_bytes = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Görüntü okunamadı"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

        # Porte algılama
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(img.shape[1] // 30, 1), 1))
        horizontal = cv2.erode(binary, horizontal_kernel)
        horizontal = cv2.dilate(horizontal, horizontal_kernel)

        h_proj = np.sum(horizontal, axis=1) / 255
        min_line_len = img.shape[1] * 0.4
        candidates = np.where(h_proj > min_line_len)[0]

        if len(candidates) == 0:
            return jsonify({"success": False, "error": "Porte bulunamadı. Daha net fotoğraf çekin."})

        # Çizgileri grupla
        groups = []
        cur = [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i-1] <= 2:
                cur.append(candidates[i])
            else:
                groups.append(int(np.mean(cur)))
                cur = [candidates[i]]
        if cur:
            groups.append(int(np.mean(cur)))

        # 5'li porte bul
        staves = []
        i = 0
        while i < len(groups) - 4:
            g = groups[i:i+5]
            spacings = [g[j+1] - g[j] for j in range(4)]
            avg_sp = np.mean(spacings)
            if avg_sp > 5 and avg_sp < 40 and all(abs(s - avg_sp) / avg_sp < 0.2 for s in spacings):
                staves.append({"lines": g, "spacing": float(avg_sp), "top": g[0], "bottom": g[4]})
                i += 5
            else:
                i += 1

        if not staves:
            return jsonify({"success": False, "error": "Porte grubu bulunamadı."})

        # Porte çizgilerini kaldır
        no_staff = binary.copy()
        for staff in staves:
            for ly in staff["lines"]:
                for dy in range(-1, 2):
                    y = ly + dy
                    if 0 <= y < img.shape[0]:
                        for x in range(img.shape[1]):
                            if no_staff[y, x] == 255:
                                top = y
                                bot = y
                                while top > 0 and no_staff[top-1, x] == 255: top -= 1
                                while bot < img.shape[0]-1 and no_staff[bot+1, x] == 255: bot += 1
                                if bot - top + 1 <= 4:
                                    no_staff[y, x] = 0

        # Nota algılama (contour)
        contours, _ = cv2.findContours(no_staff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_notes = []
        note_names_down = ['Si', 'La', 'Sol', 'Fa', 'Mi', 'Re', 'Do']

        for staff_idx, staff in enumerate(staves):
            sp = staff["spacing"]
            staff_notes = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < sp * sp * 0.3 or area > sp * sp * 8:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                aspect = w / max(h, 1)

                if cy < staff["top"] - sp * 4 or cy > staff["bottom"] + sp * 4:
                    continue

                if aspect < 0.4 or aspect > 2.0:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                fill_ratio = area / (w * h) if w * h > 0 else 0
                is_filled = fill_ratio > 0.5

                mid_line = staff["lines"][2]
                half_spaces = round((cy - mid_line) / (sp / 2))
                note_idx = half_spaces % 7
                if note_idx < 0:
                    note_idx += 7
                octave = 4
                if half_spaces > 0:
                    octave = 4 - int((half_spaces + 5) / 7)
                elif half_spaces < 0:
                    octave = 4 + int((-half_spaces + 1) / 7)
                octave = max(2, min(6, octave))

                note_name = note_names_down[note_idx % 7]

                if is_filled:
                    duration = 1
                    dur_name = "quarter"
                else:
                    duration = 2
                    dur_name = "half"

                staff_notes.append({
                    "pitch": note_name,
                    "octave": octave,
                    "duration": duration,
                    "durationName": dur_name,
                    "x": int(cx),
                    "y": int(cy),
                    "staffIndex": staff_idx,
                    "confidence": round(min(fill_ratio + 0.3, 0.95), 2)
                })

            staff_notes.sort(key=lambda n: n["x"])
            all_notes.extend(staff_notes)

        return jsonify({
            "success": True,
            "data": {
                "notes": all_notes,
                "staves": [{"index": i, "lines": s["lines"], "spacing": s["spacing"]} for i, s in enumerate(staves)],
                "metadata": {
                    "noteCount": len(all_notes),
                    "staffCount": len(staves),
                    "imageSize": {"width": img.shape[1], "height": img.shape[0]}
                }
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
