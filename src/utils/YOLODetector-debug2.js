/**
 * YOLODetector Class - Deep Debug Version
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

    console.log('🖼️ Image:', imgWidth, '×', imgHeight);
    console.log('📐 Scale:', scale.toFixed(4), '| Offset:', offsetX.toFixed(1), offsetY.toFixed(1));

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this.inputSize, this.inputSize]);
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: inputTensor });
    const output = results[this.session.outputNames[0]];

    const dims = output.dims;
    const data = output.data;

    console.log('📊 Output dims:', dims);
    console.log('📊 Data length:', data.length);

    // For [1, 84, 8400] format
    const numFeatures = dims[1];  // 84
    const numBoxes = dims[2];     // 8400
    const numClasses = numFeatures - 4;  // 80

    console.log(`📐 numBoxes=${numBoxes}, numFeatures=${numFeatures}, numClasses=${numClasses}`);

    // Deep debug: Find the maximum confidence across ALL boxes
    let globalMaxConf = 0;
    let globalMaxIdx = -1;
    let confAbove01 = 0;
    let confAbove025 = 0;
    let confAbove05 = 0;

    // Check first box's raw values
    console.log('🔢 First box (i=0) raw data:');
    console.log('  cx (data[0]):', data[0]);
    console.log('  cy (data[8400]):', data[numBoxes]);
    console.log('  w (data[16800]):', data[2 * numBoxes]);
    console.log('  h (data[25200]):', data[3 * numBoxes]);
    console.log('  class0 score (data[33600]):', data[4 * numBoxes]);
    console.log('  class1 score (data[42000]):', data[5 * numBoxes]);

    // Check middle box
    const midIdx = Math.floor(numBoxes / 2);
    console.log(`🔢 Middle box (i=${midIdx}) raw data:`);
    console.log('  cx:', data[0 * numBoxes + midIdx]);
    console.log('  cy:', data[1 * numBoxes + midIdx]);
    console.log('  w:', data[2 * numBoxes + midIdx]);
    console.log('  h:', data[3 * numBoxes + midIdx]);

    // Find max confidence across all boxes
    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numBoxes + i];
        if (score > maxScore) maxScore = score;
      }
      
      if (maxScore > 0.1) confAbove01++;
      if (maxScore > 0.25) confAbove025++;
      if (maxScore > 0.5) confAbove05++;
      
      if (maxScore > globalMaxConf) {
        globalMaxConf = maxScore;
        globalMaxIdx = i;
      }
    }

    console.log('📈 Confidence distribution:');
    console.log(`  > 0.1: ${confAbove01} boxes`);
    console.log(`  > 0.25: ${confAbove025} boxes`);
    console.log(`  > 0.5: ${confAbove05} boxes`);
    console.log(`  Max confidence: ${globalMaxConf.toFixed(4)} at box ${globalMaxIdx}`);

    // Show the best box details
    if (globalMaxIdx >= 0) {
      const i = globalMaxIdx;
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];
      
      let maxClassId = 0;
      let maxScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numBoxes + i];
        if (score > maxScore) { maxScore = score; maxClassId = c; }
      }
      
      console.log(`📦 Best box details:`);
      console.log(`  Index: ${i}`);
      console.log(`  Raw coords: cx=${cx.toFixed(2)}, cy=${cy.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}`);
      console.log(`  Class: ${maxClassId}, Confidence: ${maxScore.toFixed(4)}`);
      
      // Convert coordinates
      const x1 = ((cx - w / 2) - offsetX) / scale;
      const y1 = ((cy - h / 2) - offsetY) / scale;
      const x2 = ((cx + w / 2) - offsetX) / scale;
      const y2 = ((cy + h / 2) - offsetY) / scale;
      console.log(`  Converted: x1=${x1.toFixed(1)}, y1=${y1.toFixed(1)}, x2=${x2.toFixed(1)}, y2=${y2.toFixed(1)}`);
    }

    // Now parse all detections
    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let maxClassId = 0;

      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numBoxes + i];
        if (score > maxScore) { maxScore = score; maxClassId = c; }
      }

      if (maxScore < confThreshold) continue;

      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

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
