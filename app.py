import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
# อนุญาตให้ทุกเว็บเรียกใช้ API นี้ได้
CORS(app, resources={r"/*": {"origins": "*"}})

# โหลดโมเดล (ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ข้างๆ ไฟล์นี้)
model = YOLO('best.pt') 

@app.route("/")
def home():
    # Endpoint นี้จะถูกใช้สำหรับ Ping/Cold Start (GET request)
    return "YOLOv8 API is running!"

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # ส่งให้ AI ประมวลผล
        results = model(image)

        detections = []
        total_count = 0 # เตรียมตัวแปรนับรวม
        
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
                total_count += 1 # นับจำนวนวัตถุ

        # ส่งผลลัพธ์กลับไป (พร้อมค่า count)
        return jsonify({
            "detections": detections,
            "count": total_count # เพิ่มค่า count นี้
        })

    except Exception as e:
        # ใช้ 500 สำหรับ Internal Server Error
        return jsonify({"error": f"Internal Server Error during processing: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)