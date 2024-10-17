from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import torch
import numpy as np
from watermark_detection.utils.model import WatermarkClassifier, get_object_detection_model
from watermark_detection.utils.transforms import get_classification_transform, get_detection_transform
from watermark_detection.utils.utils import load_model
from watermark_detection.utils.predict import predict_unseen_data
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Watermark Detection API",
              description="API for detecting watermarks in images",
              version="1.0.0")

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model_path = '/Users/driandy/Desktop/Works/Personal/99 group/watermark_detection/models/classification_models/best_classification_model.pth'
detection_model_path = '/Users/driandy/Desktop/Works/Personal/99 group/watermark_detection/models/object_detection_models/final_model.pth'

classification_model = load_model(WatermarkClassifier(), classification_model_path, device)
detection_model = load_model(get_object_detection_model(num_classes=2), detection_model_path, device)

# Get transforms
classification_transform = get_classification_transform()
detection_transform = get_detection_transform()

class WatermarkDetectionResponse(BaseModel):
    watermark_probability: float
    has_watermark: bool
    detections: list

@app.post("/detect_watermark/", response_model=WatermarkDetectionResponse)
async def detect_watermark(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG, PNG, or GIF image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        logger.exception("Error reading image file")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    try:
        watermark_prob, detection_prediction = predict_unseen_data(
            classification_model, detection_model, image_np,
            classification_transform, detection_transform
        )
    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    detections = [
        {
            "box": box.tolist(),
            "score": score.item(),
            "label": "r123" if label.item() == 1 else "non_r123"
        }
        for box, score, label in zip(detection_prediction['boxes'], detection_prediction['scores'], detection_prediction['labels'])
        if score > 0.3
    ]
    
    return WatermarkDetectionResponse(
        watermark_probability=watermark_prob,
        has_watermark=watermark_prob > 0.5,
        detections=detections
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Watermark Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)