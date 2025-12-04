import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
# อนุญาตให้ทุกเว็บเรียกใช้ API นี้ได้ (แก้ * เป็นชื่อเว็บคุณถ้าต้องการความปลอดภัยสูงขึ้น)
CORS(app, resources={r"/*": {"origins": "*"}})

# โหลดโมเดล (ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ข้างๆ ไฟล์นี้)
model = YOLO('best.pt')

@app.route("/")
def home():
    return "YOLOv8 API is running!"

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        # อ่านรูปภาพ
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # ส่งให้ AI ประมวลผล
        results = model(image)

        # แกะผลลัพธ์ออกมา
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ใช้ Port จาก Environment Variable ของ Render (ถ้าไม่มีใช้ 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)