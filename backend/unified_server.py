"""
AutoLabel Pro - Unified API Server
===================================
Combined server for YOLO (v8/v11) and Grounding DINO detection

Endpoints:
- POST /yolo           - YOLOv8 detection
- POST /yolo11         - YOLOv11 detection
- POST /grounding-dino - Grounding DINO (open-world)
- GET  /health         - Health check
- GET  /models         - List available models

Usage:
    pip install fastapi uvicorn python-multipart pillow ultralytics transformers torch
    python unified_server.py

Or with uvicorn directly:
    uvicorn unified_server:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import torch
import base64
import io
import numpy as np
from collections import defaultdict

# ============================================
# FastAPI App Setup
# ============================================

app = FastAPI(
    title="AutoLabel Pro API",
    description="Unified API for YOLO and Grounding DINO object detection",
    version="1.0.0"
)

# Enable CORS for all origins (important for GitHub Pages frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global Variables
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO models
yolov8_model = None
yolov11_model = None

# Grounding DINO
gdino_processor = None
gdino_model = None

# ============================================
# Request/Response Models
# ============================================

class YOLORequest(BaseModel):
    image: str  # Base64 encoded
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

class GroundingDINORequest(BaseModel):
    image: str  # Base64 encoded
    prompt: str
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_threshold: float = 0.5

class Detection(BaseModel):
    bbox: List[float]  # [x, y, width, height]
    label: str
    confidence: float

class DetectionResponse(BaseModel):
    detections: List[Detection]
    image_size: List[int]
    model: str

class HealthResponse(BaseModel):
    status: str
    models: dict
    device: str

# ============================================
# Utility Functions
# ============================================

def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def apply_nms_per_class(boxes, scores, labels, iou_threshold=0.5):
    """Apply Per-Class Non-Maximum Suppression"""
    if len(boxes) == 0:
        return [], [], []
    
    class_detections = defaultdict(lambda: {'boxes': [], 'scores': [], 'indices': []})
    
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_detections[label]['boxes'].append(box)
        class_detections[label]['scores'].append(score)
        class_detections[label]['indices'].append(idx)
    
    keep_indices = []
    
    for label, data in class_detections.items():
        if len(data['boxes']) == 0:
            continue
        
        class_boxes = np.array(data['boxes'])
        class_scores = np.array(data['scores'])
        class_indices = np.array(data['indices'])
        
        kept = nms_single_class(class_boxes, class_scores, iou_threshold)
        keep_indices.extend(class_indices[kept].tolist())
    
    keep_indices.sort()
    
    filtered_boxes = [boxes[i] for i in keep_indices]
    filtered_scores = [scores[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    
    return filtered_boxes, filtered_scores, filtered_labels

def nms_single_class(boxes, scores, iou_threshold):
    """NMS for a single class"""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def normalize_prompt(text_prompt):
    """Normalize text prompt for Grounding DINO"""
    if not text_prompt or text_prompt.strip() == "":
        return "", []
    
    if '.' in text_prompt:
        classes = [c.strip() for c in text_prompt.split('.') if c.strip()]
    elif ',' in text_prompt:
        classes = [c.strip() for c in text_prompt.split(',') if c.strip()]
    else:
        classes = [text_prompt.strip()]
    
    if not classes:
        return "", []
    
    normalized = ". ".join(classes) + "."
    return normalized, classes

# ============================================
# Model Loaders
# ============================================

def load_yolo_models():
    """Load YOLO models"""
    global yolov8_model, yolov11_model
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics package found")
    except ImportError:
        print("❌ ultralytics not installed!")
        print("   Run: pip install ultralytics")
        return False
    
    # Load YOLOv8
    try:
        print("📦 Loading YOLOv8n...")
        yolov8_model = YOLO("yolov8n.pt")
        print("✅ YOLOv8 loaded!")
    except Exception as e:
        print(f"⚠️ Could not load YOLOv8: {e}")
        yolov8_model = None
    
    # Load YOLOv11
    try:
        print("📦 Loading YOLOv11n...")
        yolov11_model = YOLO("yolo11n.pt")
        print("✅ YOLOv11 loaded!")
    except Exception as e:
        print(f"⚠️ YOLOv11 not available: {e}")
        if yolov8_model is not None:
            yolov11_model = yolov8_model
            print("   Using YOLOv8 as fallback")
        else:
            yolov11_model = None
    
    return yolov8_model is not None or yolov11_model is not None

def load_grounding_dino():
    """Load Grounding DINO model"""
    global gdino_processor, gdino_model
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        print("📦 Loading Grounding DINO from Hugging Face...")
        print(f"🖥️ Device: {device}")
        
        model_id = "IDEA-Research/grounding-dino-tiny"
        
        gdino_processor = AutoProcessor.from_pretrained(model_id)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        
        print("✅ Grounding DINO loaded!")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not load Grounding DINO: {e}")
        return False

# ============================================
# YOLO Detection
# ============================================

def run_yolo_detection(model, image, confidence_threshold=0.25, iou_threshold=0.45):
    """Run YOLO detection"""
    if model is None:
        return []
    
    results = model(
        image,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result.names[cls_id]
            
            detections.append({
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "label": class_name,
                "confidence": confidence
            })
    
    return detections

# ============================================
# API Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models": {
            "yolov8": yolov8_model is not None,
            "yolov11": yolov11_model is not None,
            "grounding_dino": gdino_model is not None
        },
        "device": device
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": "YOLOv8",
                "endpoint": "/yolo",
                "loaded": yolov8_model is not None,
                "type": "closed-set",
                "classes": 80
            },
            {
                "name": "YOLOv11",
                "endpoint": "/yolo11",
                "loaded": yolov11_model is not None,
                "type": "closed-set",
                "classes": 80
            },
            {
                "name": "Grounding DINO",
                "endpoint": "/grounding-dino",
                "loaded": gdino_model is not None,
                "type": "open-world",
                "classes": "unlimited (text prompt)"
            }
        ]
    }

@app.post("/yolo")
async def detect_yolov8(request: YOLORequest):
    """YOLOv8 detection endpoint"""
    if yolov8_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 model not loaded")
    
    try:
        print(f"🔍 YOLOv8 | conf={request.confidence_threshold}, iou={request.iou_threshold}")
        
        image = base64_to_pil(request.image)
        image_size = image.size
        
        detections = run_yolo_detection(
            yolov8_model,
            image,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        
        print(f"✅ Found {len(detections)} objects")
        
        return {
            "detections": detections,
            "image_size": list(image_size),
            "model": "yolov8"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yolo11")
async def detect_yolov11(request: YOLORequest):
    """YOLOv11 detection endpoint"""
    if yolov11_model is None:
        raise HTTPException(status_code=503, detail="YOLOv11 model not loaded")
    
    try:
        print(f"🔍 YOLOv11 | conf={request.confidence_threshold}, iou={request.iou_threshold}")
        
        image = base64_to_pil(request.image)
        image_size = image.size
        
        detections = run_yolo_detection(
            yolov11_model,
            image,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        
        print(f"✅ Found {len(detections)} objects")
        
        return {
            "detections": detections,
            "image_size": list(image_size),
            "model": "yolov11"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grounding-dino")
async def detect_grounding_dino(request: GroundingDINORequest):
    """Grounding DINO detection endpoint"""
    if gdino_model is None:
        raise HTTPException(status_code=503, detail="Grounding DINO model not loaded")
    
    try:
        print(f"🔍 Grounding DINO | prompt='{request.prompt}'")
        print(f"   box={request.box_threshold}, text={request.text_threshold}, nms={request.nms_threshold}")
        
        image = base64_to_pil(request.image)
        image_size = image.size
        
        # Normalize prompt
        normalized_prompt, prompt_classes = normalize_prompt(request.prompt)
        if not prompt_classes:
            return {
                "detections": [],
                "image_size": list(image_size),
                "model": "grounding-dino",
                "prompt": request.prompt
            }
        
        print(f"🏷️ Classes: {prompt_classes}")
        
        # Run inference
        inputs = gdino_processor(images=image, text=normalized_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = gdino_model(**inputs)
        
        # Post-process
        try:
            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[image_size[::-1]],
                threshold=request.box_threshold
            )[0]
        except TypeError:
            results = gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
                target_sizes=[image_size[::-1]]
            )[0]
        
        # Extract detections
        raw_boxes = []
        raw_scores = []
        raw_labels = []
        
        if len(results.get('boxes', [])) > 0:
            boxes = results['boxes']
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            
            scores = results.get('scores', [])
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            
            text_labels = results.get('text_labels', None)
            labels = results.get('labels', [])
            
            class_counts = {cls: 0 for cls in prompt_classes}
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Extract label
                    label = None
                    
                    if text_labels is not None and i < len(text_labels):
                        text_label = str(text_labels[i]).strip()
                        if text_label:
                            label = text_label
                    
                    if not label and i < len(labels):
                        label_val = labels[i]
                        if isinstance(label_val, str) and label_val.strip():
                            label = label_val.strip()
                        elif isinstance(label_val, (int, np.integer)):
                            label_idx = int(label_val)
                            if 0 <= label_idx < len(prompt_classes):
                                label = prompt_classes[label_idx]
                    
                    # Fallback
                    if not label:
                        min_count = min(class_counts.values()) if class_counts else 0
                        candidates = [cls for cls, count in class_counts.items() if count == min_count]
                        label = candidates[0] if candidates else (prompt_classes[0] if prompt_classes else "object")
                    
                    if label in class_counts:
                        class_counts[label] += 1
                    
                    raw_boxes.append([float(x1), float(y1), float(width), float(height)])
                    raw_scores.append(float(score))
                    raw_labels.append(label)
        
        print(f"📦 Raw detections: {len(raw_boxes)}")
        
        # Apply Per-Class NMS
        if len(raw_boxes) > 0:
            filtered_boxes, filtered_scores, filtered_labels = apply_nms_per_class(
                raw_boxes,
                raw_scores,
                raw_labels,
                iou_threshold=request.nms_threshold
            )
            print(f"✅ After NMS: {len(filtered_boxes)} detections")
        else:
            filtered_boxes, filtered_scores, filtered_labels = [], [], []
        
        # Format response
        detections = []
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            detections.append({
                "bbox": box,
                "label": label,
                "confidence": score
            })
        
        return {
            "detections": detections,
            "image_size": list(image_size),
            "model": "grounding-dino",
            "prompt": request.prompt
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Startup
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "=" * 60)
    print("🚀 AutoLabel Pro - Unified API Server")
    print("=" * 60)
    print(f"🖥️ Device: {device}")
    print()
    
    # Load YOLO models
    print("📦 Loading YOLO models...")
    load_yolo_models()
    print()
    
    # Load Grounding DINO
    print("📦 Loading Grounding DINO...")
    load_grounding_dino()
    print()
    
    print("=" * 60)
    print("📍 Endpoints:")
    print("   POST /yolo           - YOLOv8 detection")
    print("   POST /yolo11         - YOLOv11 detection")
    print("   POST /grounding-dino - Grounding DINO")
    print("   GET  /health         - Health check")
    print("   GET  /models         - List models")
    print("=" * 60)
    print()

# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting AutoLabel Pro Unified Server...")
    print("=" * 60)
    print("📍 Server: http://localhost:8000")
    print("📍 Docs:   http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
