/**
 * GroundingDINODetector Class - Prompt-guided object detection
 * Supports both ONNX models and API endpoints
 */
class GroundingDINODetector {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.apiEndpoint = null;
    this.mode = null;
    this.tokenizer = null;
  }

  /**
   * Load ONNX model for local inference
   * Note: Grounding DINO is complex and typically requires API for best results
   */
  async loadONNXModel(modelBuffer) {
    try {
      if (typeof ort === "undefined") {
        throw new Error("ONNX Runtime not loaded");
      }
      // Note: Full Grounding DINO ONNX support requires additional setup
      // This is a placeholder for future ONNX support
      console.warn(" Grounding DINO ONNX support is experimental");
      console.log(" For best results, use API mode");

      this.session = await ort.InferenceSession.create(modelBuffer);
      this.modelLoaded = true;
      this.mode = "onnx";
      console.log(" Grounding DINO ONNX Model loaded");
      return true;
    } catch (error) {
      console.error("Failed to load Grounding DINO model:", error);
      throw error;
    }
  }

  /**
   * Set API endpoint for remote inference
   */
  setAPIEndpoint(endpoint) {
    this.apiEndpoint = endpoint;
    this.mode = "api";
    this.modelLoaded = true;
    console.log(" Grounding DINO API mode enabled:", endpoint);
  }

  /**
   * Preprocess image to base64
   */
  imageToBase64(imageElement) {
    const canvas = document.createElement("canvas");
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageElement, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.95);
  }

  /**
   * Parse text prompts (supports multiple objects separated by dots)
   * Examples:
   * - "car . person . dog"
   * - "red car . blue truck"
   * - "person wearing hat"
   */
  parsePrompts(textPrompt) {
    if (!textPrompt || textPrompt.trim() === "") {
      throw new Error("Text prompt cannot be empty");
    }

    // Split by dots and clean up
    const prompts = textPrompt
      .split(".")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    return prompts;
  }

  /**
   * Run inference via API
   */
  async runAPIInference(
    imageElement,
    textPrompt,
    boxThreshold = 0.35,
    textThreshold = 0.25
  ) {
    if (!this.apiEndpoint) {
      throw new Error("API endpoint not set");
    }

    const prompts = this.parsePrompts(textPrompt);
    console.log(" Prompts:", prompts);
    console.log(" Thresholds - Box:", boxThreshold, "Text:", textThreshold);

    const base64Image = this.imageToBase64(imageElement);

    try {
      const response = await fetch(this.apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64Image,
          prompt: textPrompt,
          box_threshold: boxThreshold,
          text_threshold: textThreshold,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log(" API Response:", data);

      // Transform API response to standard format
      return this.transformDetections(
        data,
        imageElement.width,
        imageElement.height
      );
    } catch (error) {
      console.error(" Grounding DINO API error:", error);
      throw error;
    }
  }

  /**
   * Transform API detections to standard format
   * Handles various API response formats
   */
  transformDetections(apiResponse, imgWidth, imgHeight) {
    // Handle different API response formats
    let detections = [];

    if (apiResponse.detections && Array.isArray(apiResponse.detections)) {
      detections = apiResponse.detections;
    } else if (
      apiResponse.predictions &&
      Array.isArray(apiResponse.predictions)
    ) {
      detections = apiResponse.predictions;
    } else if (Array.isArray(apiResponse)) {
      detections = apiResponse;
    }

    // Transform to standard format
    return detections.map((det, idx) => {
      // Handle different coordinate formats
      let x, y, width, height;

      if ("bbox" in det) {
        // COCO format: [x, y, width, height]
        [x, y, width, height] = det.bbox;
      } else if ("box" in det) {
        // Box format: {x, y, width, height} or [x1, y1, x2, y2]
        if (Array.isArray(det.box)) {
          const [x1, y1, x2, y2] = det.box;
          x = x1;
          y = y1;
          width = x2 - x1;
          height = y2 - y1;
        } else {
          ({ x, y, width, height } = det.box);
        }
      } else {
        // Direct format
        x = det.x || 0;
        y = det.y || 0;
        width = det.width || det.w || 0;
        height = det.height || det.h || 0;
      }

      // Handle normalized coordinates (0-1 range)
      if (x < 1 && y < 1 && width < 1 && height < 1) {
        x *= imgWidth;
        y *= imgHeight;
        width *= imgWidth;
        height *= imgHeight;
      }

      // Get label and confidence
      const label = det.label || det.class || det.className || `object_${idx}`;
      const confidence = det.confidence || det.score || 0.5;

      return {
        x: Math.max(0, x),
        y: Math.max(0, y),
        width: Math.min(imgWidth - x, width),
        height: Math.min(imgHeight - y, height),
        confidence: confidence,
        label: label,
        className: label,
      };
    });
  }

  /**
   * Main detection method
   */
  async detect(
    imageElement,
    textPrompt,
    boxThreshold = 0.35,
    textThreshold = 0.25
  ) {
    if (!this.modelLoaded) {
      throw new Error(
        "No model loaded. Please load a model or configure API endpoint."
      );
    }

    if (!textPrompt || textPrompt.trim() === "") {
      throw new Error("Text prompt is required for Grounding DINO detection");
    }

    console.log(" Starting Grounding DINO detection...");
    console.log(" Prompt:", textPrompt);

    if (this.mode === "api") {
      return await this.runAPIInference(
        imageElement,
        textPrompt,
        boxThreshold,
        textThreshold
      );
    } else if (this.mode === "onnx") {
      throw new Error(
        "ONNX mode not fully implemented yet. Please use API mode."
      );
    }

    throw new Error("Unknown detection mode");
  }

  /**
   * Batch detection on multiple images
   */
  async detectBatch(
    images,
    textPrompt,
    boxThreshold = 0.35,
    textThreshold = 0.25,
    onProgress = null
  ) {
    const results = [];

    for (let i = 0; i < images.length; i++) {
      if (onProgress) {
        onProgress(i + 1, images.length);
      }

      try {
        const detections = await this.detect(
          images[i],
          textPrompt,
          boxThreshold,
          textThreshold
        );
        results.push({ image: images[i], detections, success: true });
      } catch (error) {
        console.error(`Failed to detect on image ${i}:`, error);
        results.push({
          image: images[i],
          detections: [],
          success: false,
          error,
        });
      }
    }

    return results;
  }

  /**
   * Check if model is ready
   */
  isReady() {
    return this.modelLoaded;
  }

  /**
   * Get current mode
   */
  getMode() {
    return this.mode;
  }

  /**
   * Reset detector
   */
  reset() {
    this.session = null;
    this.modelLoaded = false;
    this.mode = null;
    this.apiEndpoint = null;
  }
}

// Create singleton instance
export const groundingDINODetector = new GroundingDINODetector();
export default GroundingDINODetector;
