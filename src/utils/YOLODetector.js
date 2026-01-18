/**
 * YOLODetector Class - Fixed for normalized coordinates
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
      if (typeof ort === "undefined") {
        throw new Error("ONNX Runtime not loaded");
      }
      this.session = await ort.InferenceSession.create(modelBuffer);
      this.modelLoaded = true;
      this.mode = "onnx";
      console.log(" ONNX Model loaded");
      console.log("Input names:", this.session.inputNames);
      console.log("Output names:", this.session.outputNames);
      return true;
    } catch (error) {
      console.error("Failed to load ONNX model:", error);
      throw error;
    }
  }

  setAPIEndpoint(endpoint) {
    this.apiEndpoint = endpoint;
    this.mode = "api";
    this.modelLoaded = true;
  }

  preprocessImage(imageElement) {
    const canvas = document.createElement("canvas");
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext("2d");

    const scale = Math.min(
      this.inputSize / imageElement.width,
      this.inputSize / imageElement.height
    );
    const scaledWidth = imageElement.width * scale;
    const scaledHeight = imageElement.height * scale;
    const offsetX = (this.inputSize - scaledWidth) / 2;
    const offsetY = (this.inputSize - scaledHeight) / 2;

    ctx.fillStyle = "#808080";
    ctx.fillRect(0, 0, this.inputSize, this.inputSize);
    ctx.drawImage(imageElement, offsetX, offsetY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const { data } = imageData;

    const float32Data = new Float32Array(3 * this.inputSize * this.inputSize);
    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      float32Data[i] = data[i * 4] / 255.0;
      float32Data[this.inputSize * this.inputSize + i] =
        data[i * 4 + 1] / 255.0;
      float32Data[2 * this.inputSize * this.inputSize + i] =
        data[i * 4 + 2] / 255.0;
    }

    return { tensor: float32Data, scale, offsetX, offsetY };
  }

  async runONNXInference(
    imageElement,
    confThreshold = 0.25,
    iouThreshold = 0.45
  ) {
    if (!this.session) throw new Error("Model not loaded");

    const { tensor, scale, offsetX, offsetY } =
      this.preprocessImage(imageElement);
    const imgWidth = imageElement.width;
    const imgHeight = imageElement.height;

    console.log(" Image:", imgWidth, "×", imgHeight);
    console.log(
      " Scale:",
      scale.toFixed(4),
      "| Offset:",
      offsetX.toFixed(1),
      offsetY.toFixed(1)
    );

    const inputTensor = new ort.Tensor("float32", tensor, [
      1,
      3,
      this.inputSize,
      this.inputSize,
    ]);
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: inputTensor });
    const output = results[this.session.outputNames[0]];

    const dims = output.dims;
    const data = output.data;

    console.log(" Output dims:", dims);

    // For [1, 84, 8400] format
    const numFeatures = dims[1]; // 84
    const numBoxes = dims[2]; // 8400
    const numClasses = numFeatures - 4; // 80

    // Detect if coordinates are normalized (values < 2 typically means normalized)
    const sampleCx = data[0];
    const isNormalized = sampleCx < 2;
    console.log(
      ` Coordinates are ${isNormalized ? "NORMALIZED (0-1)" : "PIXEL (0-640)"}`
    );

    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
      let maxScore = 0;
      let maxClassId = 0;

      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numBoxes + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = c;
        }
      }

      if (maxScore < confThreshold) continue;

      // Get raw coordinates
      let cx = data[0 * numBoxes + i];
      let cy = data[1 * numBoxes + i];
      let w = data[2 * numBoxes + i];
      let h = data[3 * numBoxes + i];

      // If normalized, convert to 640x640 pixel coordinates first
      if (isNormalized) {
        cx *= this.inputSize;
        cy *= this.inputSize;
        w *= this.inputSize;
        h *= this.inputSize;
      }

      // Convert from 640x640 to original image coordinates
      const x1 = (cx - w / 2 - offsetX) / scale;
      const y1 = (cy - h / 2 - offsetY) / scale;
      const x2 = (cx + w / 2 - offsetX) / scale;
      const y2 = (cy + h / 2 - offsetY) / scale;

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

    console.log(` Detections before NMS: ${detections.length}`);

    const finalDetections = this.nms(detections, iouThreshold);
    console.log(` Detections after NMS: ${finalDetections.length}`);

    if (finalDetections.length > 0) {
      console.log(" First detection:", finalDetections[0]);
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
    if (!this.apiEndpoint) throw new Error("API endpoint not set");

    const canvas = document.createElement("canvas");
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageElement, 0, 0);
    const base64 = canvas.toDataURL("image/jpeg", 0.9);

    const response = await fetch(this.apiEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64, confidence: confThreshold }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    return data.detections || [];
  }

  async detect(imageElement, confThreshold = 0.25, iouThreshold = 0.45) {
    if (!this.modelLoaded) throw new Error("No model loaded");

    if (this.mode === "onnx") {
      return this.runONNXInference(imageElement, confThreshold, iouThreshold);
    } else if (this.mode === "api") {
      return this.runAPIInference(imageElement, confThreshold);
    }

    throw new Error("Unknown detection mode");
  }
}

export const detector = new YOLODetector();
export default YOLODetector;
