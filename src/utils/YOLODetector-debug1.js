/**
 * YOLODetector Class - Debug Version
 * Tests both possible YOLOv8 output layouts
 */
class YOLODetector {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.inputSize = 640;
    this.apiEndpoint = null;
    this.mode = null;
  }

  async loadONNXModel(modelBuffer) {
    try {
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime not loaded');
      }
      this.session = await ort.InferenceSession.create(modelBuffer);
      this.modelLoaded = true;
      this.mode = 'onnx';
      console.log('✅ ONNX Model loaded');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  setAPIEndpoint(endpoint) {
    this.apiEndpoint = endpoint;
    this.mode = 'api';
    this.modelLoaded = true;
  }

  preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext('2d');

    const scale = Math.min(
      this.inputSize / imageElement.width,
      this.inputSize / imageElement.height
    );
    const scaledWidth = imageElement.width * scale;
    const scaledHeight = imageElement.height * scale;
    const offsetX = (this.inputSize - scaledWidth) / 2;
    const offsetY = (this.inputSize - scaledHeight) / 2;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, this.inputSize, this.inputSize);
    ctx.drawImage(imageElement, offsetX, offsetY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const { data } = imageData;

    const float32Data = new Float32Array(3 * this.inputSize * this.inputSize);
    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      float32Data[i] = data[i * 4] / 255.0;
      float32Data[this.inputSize * this.inputSize + i] = data[i * 4 + 1] / 255.0;
      float32Data[2 * this.inputSize * this.inputSize + i] = data[i * 4 + 2] / 255.0;
    }

    return { tensor: float32Data, scale, offsetX, offsetY };
  }

  async runONNXInference(imageElement, confThreshold = 0.25, iouThreshold = 0.45) {
    if (!this.session) throw new Error('Model not loaded');

    const { tensor, scale, offsetX, offsetY } = this.preprocessImage(imageElement);
    const imgWidth = imageElement.width;
    const imgHeight = imageElement.height;

    console.log('🖼️ Image:', imgWidth, '×', imgHeight, '| Scale:', scale.toFixed(4), '| Offset:', offsetX.toFixed(1), offsetY.toFixed(1));

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this.inputSize, this.inputSize]);
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: inputTensor });
    const output = results[this.session.outputNames[0]];

    const dims = output.dims;
    const data = output.data;

    console.log('📊 Output dims:', dims);
    console.log('📊 Data length:', data.length);
    console.log('📊 Sample values [0-9]:', Array.from(data.slice(0, 10)).map(v => v.toFixed(3)));
    console.log('📊 Sample values [100-109]:', Array.from(data.slice(100, 110)).map(v => v.toFixed(3)));

    // Detect format: [1, 84, 8400] vs [1, 8400, 84]
    let numBoxes, numFeatures, transposed;

    if (dims.length === 3) {
      if (dims[1] < dims[2]) {
        // [1, 84, 8400]
        numFeatures = dims[1];
        numBoxes = dims[2];
        transposed = false;
        console.log('📐 Format: [1, features, boxes] =', dims);
      } else {
        // [1, 8400, 84]
        numBoxes = dims[1];
        numFeatures = dims[2];
        transposed = true;
        console.log('📐 Format: [1, boxes, features] =', dims);
      }
    } else if (dims.length === 2) {
      numBoxes = dims[0];
      numFeatures = dims[1];
      transposed = true;
      console.log('📐 Format: [boxes, features] =', dims);
    }

    const numClasses = numFeatures - 4;
    console.log(`📐 numBoxes=${numBoxes}, numFeatures=${numFeatures}, numClasses=${numClasses}, transposed=${transposed}`);

    // Try to find boxes with high confidence first (debug)
    console.log('🔍 Searching for high-confidence detections...');
    
    let highConfCount = 0;
    for (let i = 0; i < Math.min(100, numBoxes); i++) {
      let maxScore = 0;
      
      if (transposed) {
        for (let c = 0; c < numClasses; c++) {
          const score = data[i * numFeatures + 4 + c];
          if (score > maxScore) maxScore = score;
        }
      } else {
        for (let c = 0; c < numClasses; c++) {
          const score = data[(4 + c) * numBoxes + i];
          if (score > maxScore) maxScore = score;
        }
      }
      
      if (maxScore > 0.5) {
        highConfCount++;
        if (highConfCount <= 3) {
          let cx, cy, w, h;
          if (transposed) {
            cx = data[i * numFeatures + 0];
            cy = data[i * numFeatures + 1];
            w = data[i * numFeatures + 2];
            h = data[i * numFeatures + 3];
          } else {
            cx = data[0 * numBoxes + i];
            cy = data[1 * numBoxes + i];
            w = data[2 * numBoxes + i];
            h = data[3 * numBoxes + i];
          }
          console.log(`  📦 Box ${i}: conf=${maxScore.toFixed(3)}, cx=${cx.toFixed(1)}, cy=${cy.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}`);
        }
      }
    }
    console.log(`🔍 Found ${highConfCount} boxes with conf > 0.5 in first 100 boxes`);

    // Now parse all detections
    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
      let cx, cy, w, h, maxScore = 0, maxClassId = 0;

      if (transposed) {
        const base = i * numFeatures;
        cx = data[base + 0];
        cy = data[base + 1];
        w = data[base + 2];
        h = data[base + 3];
        for (let c = 0; c < numClasses; c++) {
          const score = data[base + 4 + c];
          if (score > maxScore) { maxScore = score; maxClassId = c; }
        }
      } else {
        cx = data[0 * numBoxes + i];
        cy = data[1 * numBoxes + i];
        w = data[2 * numBoxes + i];
        h = data[3 * numBoxes + i];
        for (let c = 0; c < numClasses; c++) {
          const score = data[(4 + c) * numBoxes + i];
          if (score > maxScore) { maxScore = score; maxClassId = c; }
        }
      }

      if (maxScore < confThreshold) continue;

      // Convert 640x640 coords to original image coords
      const x1 = ((cx - w / 2) - offsetX) / scale;
      const y1 = ((cy - h / 2) - offsetY) / scale;
      const x2 = ((cx + w / 2) - offsetX) / scale;
      const y2 = ((cy + h / 2) - offsetY) / scale;

      const finalX = Math.max(0, x1);
      const finalY = Math.max(0, y1);
      const finalW = Math.min(imgWidth, x2) - finalX;
      const finalH = Math.min(imgHeight, y2) - finalY;

      if (finalW > 5 && finalH > 5) {
        detections.push({
          x: finalX,
          y: finalY,
          width: finalW,
          height: finalH,
          confidence: maxScore,
          classId: maxClassId,
        });
      }
    }

    console.log(`✅ Detections before NMS: ${detections.length}`);

    const finalDetections = this.nms(detections, iouThreshold);
    console.log(`✅ Detections after NMS: ${finalDetections.length}`);

    if (finalDetections.length > 0) {
      console.log('📦 First detection:', finalDetections[0]);
    }

    return finalDetections;
  }

  nms(detections, iouThreshold) {
    detections.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    const suppressed = new Set();

    for (let i = 0; i < detections.length; i++) {
      if (suppressed.has(i)) continue;
      kept.push(detections[i]);

      for (let j = i + 1; j < detections.length; j++) {
        if (suppressed.has(j)) continue;
        if (detections[i].classId !== detections[j].classId) continue;

        const iou = this.calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) suppressed.add(j);
      }
    }
    return kept;
  }

  calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;

    return union > 0 ? intersection / union : 0;
  }

  async runAPIInference(imageElement, confThreshold = 0.25) {
    if (!this.apiEndpoint) throw new Error('API endpoint not set');

    const canvas = document.createElement('canvas');
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.9);

    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64, confidence: confThreshold }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    return data.detections || [];
  }

  async detect(imageElement, confThreshold = 0.25, iouThreshold = 0.45) {
    if (!this.modelLoaded) throw new Error('No model loaded');

    if (this.mode === 'onnx') {
      return this.runONNXInference(imageElement, confThreshold, iouThreshold);
    } else if (this.mode === 'api') {
      return this.runAPIInference(imageElement, confThreshold);
    }

    throw new Error('Unknown detection mode');
  }
}

export const detector = new YOLODetector();
export default YOLODetector;
