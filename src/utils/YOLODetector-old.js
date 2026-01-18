/**
 * YOLODetector Class
 * 處理 ONNX 模型和 API 兩種推論模式
 */
class YOLODetector {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.inputSize = 640;
    this.apiEndpoint = null;
    this.mode = null; // 'onnx' or 'api'
  }

  /**
   * 載入 ONNX 模型（瀏覽器端推論）
   */
  async loadONNXModel(modelBuffer) {
    try {
      // 注意：需要在 index.html 中載入 ort (ONNX Runtime)
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime not loaded. Please add ort script to index.html');
      }
      this.session = await ort.InferenceSession.create(modelBuffer);
      this.modelLoaded = true;
      this.mode = 'onnx';
      console.log('ONNX Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * 設定 API 端點（伺服器端推論）
   */
  setAPIEndpoint(endpoint) {
    this.apiEndpoint = endpoint;
    this.mode = 'api';
    this.modelLoaded = true;
  }

  /**
   * 影像預處理
   * 將圖片縮放並正規化為 YOLO 輸入格式
   */
  preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext('2d');

    // 計算縮放比例（保持長寬比）
    const scale = Math.min(
      this.inputSize / imageElement.width,
      this.inputSize / imageElement.height
    );
    const scaledWidth = imageElement.width * scale;
    const scaledHeight = imageElement.height * scale;
    const offsetX = (this.inputSize - scaledWidth) / 2;
    const offsetY = (this.inputSize - scaledHeight) / 2;

    // 填充灰色背景並繪製圖片
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, this.inputSize, this.inputSize);
    ctx.drawImage(imageElement, offsetX, offsetY, scaledWidth, scaledHeight);

    // 取得像素資料並轉換為 tensor 格式
    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const { data } = imageData;

    // 轉換為 CHW 格式並正規化到 [0, 1]
    const float32Data = new Float32Array(3 * this.inputSize * this.inputSize);
    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      float32Data[i] = data[i * 4] / 255.0; // R
      float32Data[this.inputSize * this.inputSize + i] = data[i * 4 + 1] / 255.0; // G
      float32Data[2 * this.inputSize * this.inputSize + i] = data[i * 4 + 2] / 255.0; // B
    }

    return {
      tensor: float32Data,
      scale,
      offsetX,
      offsetY,
    };
  }

  /**
   * 執行 ONNX 推論
   */
  async runONNXInference(imageElement, confThreshold = 0.25, iouThreshold = 0.45) {
    if (!this.session) throw new Error('Model not loaded');

    const { tensor, scale, offsetX, offsetY } = this.preprocessImage(imageElement);

    // 建立 ONNX tensor
    const inputTensor = new ort.Tensor(
      'float32',
      tensor,
      [1, 3, this.inputSize, this.inputSize]
    );

    // 執行推論
    const inputName = this.session.inputNames[0];
    const results = await this.session.run({ [inputName]: inputTensor });

    // 取得輸出（YOLOv8 格式: [1, 84, 8400] 或類似）
    const output = results[this.session.outputNames[0]];

    // 解析偵測結果
    return this.parseYOLOv8Output(
      output,
      imageElement.width,
      imageElement.height,
      scale,
      offsetX,
      offsetY,
      confThreshold,
      iouThreshold
    );
  }

  /**
   * 解析 YOLOv8 輸出格式
   */
  parseYOLOv8Output(output, imgWidth, imgHeight, scale, offsetX, offsetY, confThreshold, iouThreshold) {
    const data = output.data;
    const [batch, features, numBoxes] = output.dims;
    const numClasses = features - 4; // 前 4 個是 x, y, w, h

    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
      // 取得類別分數
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

      // 取得 bounding box 座標（中心 x, 中心 y, 寬, 高）
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

      // 轉換為原始圖片座標
      const x1 = ((cx - w / 2) - offsetX) / scale;
      const y1 = ((cy - h / 2) - offsetY) / scale;
      const x2 = ((cx + w / 2) - offsetX) / scale;
      const y2 = ((cy + h / 2) - offsetY) / scale;

      // 裁切到圖片範圍內
      const clippedX1 = Math.max(0, Math.min(imgWidth, x1));
      const clippedY1 = Math.max(0, Math.min(imgHeight, y1));
      const clippedX2 = Math.max(0, Math.min(imgWidth, x2));
      const clippedY2 = Math.max(0, Math.min(imgHeight, y2));

      detections.push({
        x: clippedX1,
        y: clippedY1,
        width: clippedX2 - clippedX1,
        height: clippedY2 - clippedY1,
        confidence: maxScore,
        classId: maxClassId,
      });
    }

    // 套用 NMS
    return this.nms(detections, iouThreshold);
  }

  /**
   * Non-Maximum Suppression (非極大值抑制)
   * 移除重疊的偵測框，保留信心度最高的
   */
  nms(detections, iouThreshold) {
    // 按信心度排序（高到低）
    detections.sort((a, b) => b.confidence - a.confidence);

    const kept = [];
    const suppressed = new Set();

    for (let i = 0; i < detections.length; i++) {
      if (suppressed.has(i)) continue;

      kept.push(detections[i]);

      for (let j = i + 1; j < detections.length; j++) {
        if (suppressed.has(j)) continue;

        const iou = this.calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          suppressed.add(j);
        }
      }
    }

    return kept;
  }

  /**
   * 計算兩個 bounding box 的 IoU (Intersection over Union)
   */
  calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  /**
   * 執行 API 推論
   */
  async runAPIInference(imageElement, confThreshold = 0.25) {
    if (!this.apiEndpoint) throw new Error('API endpoint not set');

    // 將圖片轉換為 base64
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.9);

    // 送到 API
    const response = await fetch(this.apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64,
        confidence: confThreshold,
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    return data.detections || [];
  }

  /**
   * 主要偵測函式
   */
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

// 建立全域偵測器實例
export const detector = new YOLODetector();
export default YOLODetector;
