"""
YOLOv8/v11 API Server for AutoLabel Pro

Compatible with the AutoLabel Pro frontend.
Supports both YOLOv8 and YOLOv11 models.

Requirements:
pip install flask flask-cors pillow ultralytics

Usage:
python yolo_server.py

Endpoints:
- POST /yolo - Run YOLOv8 detection
- POST /yolo11 - Run YOLOv11 detection  
- GET /health - Health check

IMPORTANT: This server has CORS enabled to allow cross-origin requests
from the frontend running on a different port/domain.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Enable CORS for all routes and all origins
# This is essential for the frontend to connect from a different port
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Global models
yolov8_model = None
yolov11_model = None

def load_models():
    """Load YOLO models"""
    global yolov8_model, yolov11_model
    
    # First check if ultralytics is installed
    try:
        from ultralytics import YOLO
        print(" ultralytics package found")
    except ImportError:
        print(" ultralytics not installed!")
        print("   Run: pip install ultralytics")
        print("   Then restart this server.")
        return False
    
    # Try to load YOLOv8
    try:
        print(" Loading YOLOv8n (this may download weights on first run)...")
        yolov8_model = YOLO("yolov8n.pt")  # Will auto-download if not present
        print(" YOLOv8 loaded successfully!")
        print(f"   Model: {yolov8_model.model_name if hasattr(yolov8_model, 'model_name') else 'yolov8n'}")
    except Exception as e:
        print(f" Could not load YOLOv8: {e}")
        print("   Make sure you have internet access for first download")
        yolov8_model = None
    
    # Try to load YOLOv11 (uses same API)
    try:
        print(" Loading YOLOv11n...")
        yolov11_model = YOLO("yolo11n.pt")  # YOLOv11 model name
        print(" YOLOv11 loaded successfully!")
    except Exception as e:
        print(f" YOLOv11 not available: {e}")
        # Fallback: use YOLOv8 for v11 endpoint
        if yolov8_model is not None:
            yolov11_model = yolov8_model
            print(" Using YOLOv8 as fallback for YOLOv11 endpoint")
        else:
            yolov11_model = None
    
    # Return True if at least one model loaded
    if yolov8_model is not None or yolov11_model is not None:
        return True
    else:
        print("\n No models could be loaded!")
        print("   Troubleshooting:")
        print("   1. Install ultralytics: pip install ultralytics")
        print("   2. Make sure you have internet for first model download")
        print("   3. Check disk space for model weights (~6MB)")
        return False

def base64_to_pil(base64_string):
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def run_yolo_detection(model, image, confidence_threshold=0.25, iou_threshold=0.45):
    """
    Run YOLO detection on image
    
    Args:
        model: YOLO model
        image: PIL Image
        confidence_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of detections with format:
        [{"bbox": [x, y, w, h], "label": "car", "confidence": 0.92}, ...]
    """
    if model is None:
        return []
    
    # Run inference
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
            # Get bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Convert to [x, y, width, height] format
            width = x2 - x1
            height = y2 - y1
            
            # Get class and confidence
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            # Get class name
            class_name = result.names[cls_id]
            
            detections.append({
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "label": class_name,
                "confidence": confidence,
                "class_id": cls_id
            })
    
    return detections

@app.route('/yolo', methods=['POST'])
def detect_yolov8():
    """YOLOv8 detection endpoint"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Parse parameters
        confidence = data.get('confidence_threshold', 0.25)
        iou = data.get('iou_threshold', 0.45)
        
        print(f"🔍 YOLOv8 detection | conf={confidence}, iou={iou}")
        
        # Convert image
        image = base64_to_pil(data['image'])
        image_size = image.size
        print(f"   Image size: {image_size}")
        
        # Run detection
        if yolov8_model is None:
            return jsonify({
                "error": "YOLOv8 model not loaded. Check server console for details. Make sure 'ultralytics' is installed: pip install ultralytics"
            }), 500
        
        detections = run_yolo_detection(
            yolov8_model,
            image,
            confidence_threshold=confidence,
            iou_threshold=iou
        )
        
        print(f"✅ Found {len(detections)} objects")
        
        return jsonify({
            "detections": detections,
            "image_size": list(image_size),
            "model": "yolov8",
            "confidence_threshold": confidence,
            "iou_threshold": iou
        })
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/yolo11', methods=['POST'])
def detect_yolov11():
    """YOLOv11 detection endpoint"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        confidence = data.get('confidence_threshold', 0.25)
        iou = data.get('iou_threshold', 0.45)
        
        print(f" YOLOv11 detection | conf={confidence}, iou={iou}")
        
        image = base64_to_pil(data['image'])
        image_size = image.size
        print(f"   Image size: {image_size}")
        
        if yolov11_model is None:
            return jsonify({
                "error": "YOLOv11 model not loaded. Check server console for details. Make sure 'ultralytics' is installed: pip install ultralytics"
            }), 500
        
        detections = run_yolo_detection(
            yolov11_model,
            image,
            confidence_threshold=confidence,
            iou_threshold=iou
        )
        
        print(f" Found {len(detections)} objects")
        
        return jsonify({
            "detections": detections,
            "image_size": list(image_size),
            "model": "yolov11",
            "confidence_threshold": confidence,
            "iou_threshold": iou
        })
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "models": {
            "yolov8": yolov8_model is not None,
            "yolov11": yolov11_model is not None
        },
        "endpoints": {
            "/yolo": "YOLOv8 detection",
            "/yolo11": "YOLOv11 detection",
            "/health": "Health check"
        }
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "available_models": [
            {
                "name": "YOLOv8",
                "endpoint": "/yolo",
                "loaded": yolov8_model is not None,
                "classes": 80,
                "type": "closed-set"
            },
            {
                "name": "YOLOv11", 
                "endpoint": "/yolo11",
                "loaded": yolov11_model is not None,
                "classes": 80,
                "type": "closed-set"
            }
        ]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" YOLO API Server for AutoLabel Pro")
    print("="*60)
    
    # Load models
    print("\n Loading models...")
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n" + "!"*60)
        print("⚠️  WARNING: No models loaded!")
        print("!"*60)
        print("\nThe server will start but detection will fail.")
        print("To fix this, run these commands:")
        print("  pip install ultralytics")
        print("  python yolo_server.py  # restart server")
        print("!"*60)
    else:
        print("\n Models ready!")
    
    print("\n📍 Endpoints:")
    print("   POST /yolo     - YOLOv8 detection")
    print("   POST /yolo11   - YOLOv11 detection")
    print("   GET  /health   - Health check")
    print("   GET  /models   - List available models")
    print("\n" + "="*60)
    print(" Server starting on http://0.0.0.0:8001")
    print("   Local:  http://localhost:8001")
    print("   Network: http://<your-ip>:8001")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8001, debug=False)
