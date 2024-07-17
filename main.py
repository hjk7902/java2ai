from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

app = FastAPI()

model = YOLO('yolov8n.pt') # YOLOv8 모델 로드


# 데이터 모델 정의
class DetectionResult(BaseModel):
    message: str
    image: str


def detect_objects(image: Image.Image):
    # 이미지를numpy 배열로 변환
    img = np.array(image)
    results = model(img)

    # 결과를 바운딩 박스와 정확도로 이미지에 표시
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 결과 이미지를PIL로 변환
    result_image = Image.fromarray(img)
    return result_image


@app.get("/")
async def index():
    return {"message": "Hello FastAPI"}


@app.post("/detect", response_model=DetectionResult)
async def detect_service(message: str = Form(...), file: UploadFile = File(...)):
    # 이미지를 읽어서PIL 이미지로 변환
    image = Image.open(io.BytesIO(await file.read()))

    # 객체 탐지 수행
    result_image = detect_objects(image)

    # 이미지 결과를base64로 인코딩
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return DetectionResult(message=message, image=img_str)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)