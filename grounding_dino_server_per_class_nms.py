"""
Grounding DINO API Server with Per-Class NMS
Applies NMS separately for each class to avoid cross-class suppression

Requirements:
pip install flask flask-cors pillow transformers torch torchvision

IMPORTANT: This server has CORS enabled to allow cross-origin requests
from the frontend running on a different port/domain.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import base64
import io
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Enable CORS for all routes and all origins
# This is essential for the frontend to connect from a different port
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Global model and processor
processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model():
    """Initialize Grounding DINO model from Hugging Face"""
    global processor, model
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        print("🔄 Loading Grounding DINO from Hugging Face...")
        print(f"🖥️ Using device: {device}")
        
        model_id = "IDEA-Research/grounding-dino-tiny"
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("🎨 Will use demo mode instead")
        return False

def apply_nms_per_class(boxes, scores, labels, iou_threshold=0.5):
    """
    Apply Per-Class Non-Maximum Suppression
    NMS is applied separately for each class to avoid cross-class suppression
    
    Args:
        boxes: List of bounding boxes [x, y, width, height]
        scores: List of confidence scores
        labels: List of labels
        iou_threshold: IoU threshold for NMS (default 0.5)
    
    Returns:
        Filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return [], [], []
    
    # Group detections by class
    class_detections = defaultdict(lambda: {'boxes': [], 'scores': [], 'indices': []})
    
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_detections[label]['boxes'].append(box)
        class_detections[label]['scores'].append(score)
        class_detections[label]['indices'].append(idx)
    
    print(f"📊 Detections by class:")
    for label, data in class_detections.items():
        print(f"   {label}: {len(data['boxes'])} boxes")
    
    # Apply NMS per class
    keep_indices = []
    
    for label, data in class_detections.items():
        if len(data['boxes']) == 0:
            continue
        
        # Convert to numpy
        class_boxes = np.array(data['boxes'])
        class_scores = np.array(data['scores'])
        class_indices = np.array(data['indices'])
        
        # Apply NMS for this class only
        kept = nms_single_class(class_boxes, class_scores, iou_threshold)
        
        # Add kept indices
        keep_indices.extend(class_indices[kept].tolist())
        
        print(f"   {label}: kept {len(kept)}/{len(class_boxes)} boxes after NMS")
    
    # Sort by original order
    keep_indices.sort()
    
    # Return filtered results
    filtered_boxes = [boxes[i] for i in keep_indices]
    filtered_scores = [scores[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    
    return filtered_boxes, filtered_scores, filtered_labels

def nms_single_class(boxes, scores, iou_threshold):
    """
    NMS for a single class
    
    Args:
        boxes: numpy array of shape (N, 4) with format [x, y, w, h]
        scores: numpy array of shape (N,)
        iou_threshold: IoU threshold
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Pick the detection with highest score
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def base64_to_pil(base64_string):
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def normalize_prompt(text_prompt):
    """
    Normalize text prompt for Grounding DINO.
    
    Grounding DINO expects each class to end with a period.
    This function handles various input formats:
    - "car . person" -> "car. person."
    - "car, person" -> "car. person."
    - "car person" -> "car. person."
    - "car. person." -> "car. person." (already correct)
    
    Returns:
        tuple: (normalized_prompt, list of class names)
    """
    if not text_prompt or text_prompt.strip() == "":
        return "", []
    
    # Split by common delimiters: period, comma, or multiple spaces
    # First, try splitting by periods (standard format)
    if '.' in text_prompt:
        classes = [c.strip() for c in text_prompt.split('.') if c.strip()]
    elif ',' in text_prompt:
        classes = [c.strip() for c in text_prompt.split(',') if c.strip()]
    else:
        # Single class or space-separated (less common)
        classes = [text_prompt.strip()]
    
    if not classes:
        return "", []
    
    # Build normalized prompt: "class1. class2. class3."
    # Each class must end with a period for Grounding DINO
    normalized = ". ".join(classes) + "."
    
    print(f"🔄 Prompt normalization:")
    print(f"   Input:  '{text_prompt}'")
    print(f"   Output: '{normalized}'")
    print(f"   Classes: {classes}")
    
    return normalized, classes

@app.route('/grounding-dino', methods=['POST'])
def detect():
    """Grounding DINO detection endpoint with Per-Class NMS"""
    try:
        data = request.json
        
        # Extract parameters
        image_b64 = data.get('image')
        text_prompt = data.get('prompt', '')
        box_threshold = data.get('box_threshold', 0.35)
        text_threshold = data.get('text_threshold', 0.25)
        nms_threshold = data.get('nms_threshold', 0.5)
        
        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400
        
        if not text_prompt:
            return jsonify({'error': 'No text prompt provided'}), 400
        
        # Convert base64 to PIL Image
        image = base64_to_pil(image_b64)
        image_size = image.size
        
        print(f"📝 Original Prompt: {text_prompt}")
        print(f"🖼️ Image size: {image_size}")
        print(f"📊 Thresholds - Box: {box_threshold}, Text: {text_threshold}, NMS: {nms_threshold}")
        
        # Normalize the prompt for Grounding DINO
        normalized_prompt, prompt_classes = normalize_prompt(text_prompt)
        
        if not normalized_prompt:
            return jsonify({'error': 'Invalid text prompt'}), 400
        
        # Run detection
        if model is not None and processor is not None:
            print("🔍 Running Grounding DINO inference...")
            
            # Prepare inputs with NORMALIZED prompt
            inputs = processor(
                images=image,
                text=normalized_prompt,  # Use normalized prompt!
                return_tensors="pt"
            ).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-process results
            try:
                # Try new API first
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    target_sizes=[image_size[::-1]],
                    threshold=box_threshold
                )[0]
                print("✅ Using new API (threshold parameter)")
            except TypeError:
                try:
                    # Fall back to old API
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        target_sizes=[image_size[::-1]]
                    )[0]
                    print("✅ Using old API (box_threshold/text_threshold)")
                except Exception as e:
                    print(f"⚠️ Post-processing error: {e}")
                    return jsonify({'error': str(e)}), 500
            
            # Extract raw detections
            raw_boxes = []
            raw_scores = []
            raw_labels = []
            
            if len(results.get('boxes', [])) > 0:
                boxes = results['boxes']
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu().numpy()
                elif not isinstance(boxes, np.ndarray):
                    boxes = np.array(boxes)
                
                scores = results.get('scores', [])
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                elif not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
                
                # prompt_classes already extracted from normalize_prompt()
                print(f"🏷️ Prompt classes: {prompt_classes}")
                
                # Try to get labels - check what format they're in
                text_labels = results.get('text_labels', None)
                labels = results.get('labels', [])
                
                # Debug: Check what we got
                if text_labels is not None:
                    print(f"✅ text_labels found: {text_labels}")
                else:
                    print(f"⚠️ text_labels not found, using labels")
                    if len(labels) > 0:
                        print(f"📋 labels type: {type(labels[0])}, first few: {labels[:3]}")
                
                # Count how many of each class we've seen (for smart empty label assignment)
                class_counts = {cls: 0 for cls in prompt_classes}
                
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Extract label
                        label = None
                        
                        # Method 1: text_labels (string labels) - but only if not empty!
                        if text_labels is not None and i < len(text_labels):
                            text_label = str(text_labels[i]).strip()
                            if text_label:  # Not empty
                                label = text_label
                                if i < 3:  # Debug first 3
                                    print(f"  Box {i}: text_label='{label}'")
                        
                        # Method 2: If text_label was empty, use integer labels
                        if not label and i < len(labels):
                            label_val = labels[i]
                            
                            if isinstance(label_val, str) and label_val.strip():
                                # String label
                                label = label_val.strip()
                                print(f"  Box {i}: string label='{label}'")
                                    
                            elif isinstance(label_val, (int, np.integer)):
                                # Integer label - map to prompt class
                                label_idx = int(label_val)
                                if 0 <= label_idx < len(prompt_classes):
                                    label = prompt_classes[label_idx]
                                    print(f"  Box {i}: int label={label_idx} → '{label}'")  # Print for ALL
                                else:
                                    # Index out of range - use modulo to wrap around
                                    label_idx = label_idx % len(prompt_classes) if prompt_classes else 0
                                    label = prompt_classes[label_idx] if prompt_classes else "detected_object"
                                    print(f"  Box {i}: int label OUT OF RANGE, wrapped to {label_idx} → '{label}'")
                            else:
                                # Unknown type - try converting to int
                                try:
                                    label_idx = int(label_val)
                                    if 0 <= label_idx < len(prompt_classes):
                                        label = prompt_classes[label_idx]
                                    print(f"  Box {i}: converted to int={label_idx} → '{label}'")
                                except:
                                    print(f"  Box {i}: unknown label type={type(label_val)}, value={label_val}")
                        
                        # Fallback: Smart assignment for empty labels
                        if not label:
                            # Find the class with fewest detections
                            min_count = min(class_counts.values()) if class_counts else 0
                            candidates = [cls for cls, count in class_counts.items() if count == min_count]
                            label = candidates[0] if candidates else (prompt_classes[0] if prompt_classes else "detected_object")
                            print(f"  Box {i}: EMPTY LABEL → smart fallback to '{label}' (counts: {class_counts})")
                        
                        # Update class count
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
                    iou_threshold=nms_threshold
                )
                print(f"✅ After Per-Class NMS: {len(filtered_boxes)} detections")
            else:
                filtered_boxes, filtered_scores, filtered_labels = [], [], []
            
            # Format final detections
            detections = []
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                detections.append({
                    'bbox': box,
                    'label': label,
                    'confidence': score
                })
            
        else:
            print("⚠️ Model not loaded")
            detections = []
        
        return jsonify({
            'detections': detections,
            'prompt': text_prompt,
            'image_size': list(image_size),
            'model_mode': 'real' if model is not None else 'demo',
            'device': device if model is not None else 'cpu',
            'nms_applied': True,
            'nms_type': 'per-class',
            'nms_threshold': nms_threshold
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': device,
        'model_name': 'grounding-dino-tiny' if model is not None else 'none',
        'nms_enabled': True,
        'nms_type': 'per-class'
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available model variants"""
    return jsonify({
        'available_models': [
            {
                'name': 'grounding-dino-tiny',
                'id': 'IDEA-Research/grounding-dino-tiny',
                'speed': 'fast',
                'accuracy': 'good'
            }
        ],
        'features': ['Per-Class NMS for multi-class detection']
    })

if __name__ == '__main__':
    # Initialize model at startup
    model_loaded = initialize_model()
    
    if not model_loaded:
        print("⚠️ Running in DEMO MODE")
    
    # Run server
    print("\n" + "="*60)
    print("🚀 Grounding DINO API Server with Per-Class NMS")
    print("="*60)
    print(f"📍 Detection endpoint: http://localhost:8000/grounding-dino")
    print(f"💚 Health check: http://localhost:8000/health")
    print(f"📋 Available models: http://localhost:8000/models")
    print(f"✨ Features: Per-Class NMS (cars don't suppress persons!)")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
