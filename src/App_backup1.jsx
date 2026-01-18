import React, { useState, useEffect, useCallback, useRef } from "react";

/**
 * AutoLabel Pro v6.0 — Complete Version
 *
 * Restored Features:
 * - Workflow stepper (1 Import → 2 Detect → 3 Review → 4 Export)
 * - Approve, Flag, Delete buttons in right panel
 * - Larger prompt textarea with hint
 * - Sensitivity/Threshold controls moved to right panel
 *
 * New Features:
 * - Multiple export formats (YOLO, COCO, Pascal VOC, JSON)
 * - Zoom in/out for annotation
 * - Per-model server URL configuration
 * - Better CORS error handling
 */

// ===== MODEL CONFIGURATIONS =====
const MODEL_CONFIGS = {
  "yolo-v8": {
    id: "yolo-v8",
    name: "YOLOv8",
    category: "Closed-Set Detection",
    requiresPrompt: false,
    endpoint: "/yolo",
    healthEndpoint: "/health",
    defaultPort: 8001,
    thresholds: {
      confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 },
      iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 },
    },
  },
  "yolo-v11": {
    id: "yolo-v11",
    name: "YOLOv11",
    category: "Closed-Set Detection",
    requiresPrompt: false,
    endpoint: "/yolo11",
    healthEndpoint: "/health",
    defaultPort: 8001,
    thresholds: {
      confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 },
      iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 },
    },
  },
  "grounding-dino": {
    id: "grounding-dino",
    // name: "Grounding DINO",
    name: "VLM Model",
    category: "Open-World Detection",
    requiresPrompt: true,
    endpoint: "/grounding-dino",
    healthEndpoint: "/health",
    defaultPort: 8000,
    thresholds: {
      box: { label: "Box Threshold", default: 0.35, min: 0.1, max: 0.9 },
      text: { label: "Text Threshold", default: 0.25, min: 0.1, max: 0.9 },
      nms: { label: "NMS Threshold", default: 0.5, min: 0.1, max: 0.9 },
    },
  },
  "coming-soon": {
    id: "yolo-v11",
    name: "Coming Soon",
    category: "Open-World Detection",
    requiresPrompt: false,
    endpoint: "/yolo11",
    healthEndpoint: "/health",
    defaultPort: 8001,
    thresholds: {
      confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 },
      iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 },
    },
  },
};

// Export format options
const EXPORT_FORMATS = {
  yolo: {
    name: "YOLO",
    ext: "txt",
    description: "class x_center y_center width height (normalized)",
  },
  coco: {
    name: "COCO JSON",
    ext: "json",
    description: "Standard COCO dataset format",
  },
  voc: { name: "Pascal VOC", ext: "xml", description: "XML annotation format" },
  json: { name: "JSON", ext: "json", description: "Simple JSON format" },
};

// Sensitivity presets
const SENSITIVITY_PRESETS = {
  strict: {
    label: "Strict",
    // icon: "🎯",
    multiplier: 1.4,
    desc: "Fewer, confident",
  },
  balanced: {
    label: "Balanced",
    // icon: "⚖️",
    multiplier: 1.0,
    desc: "Default",
  },
  loose: {
    label: "Loose",
    // icon: "🔍",
    multiplier: 0.7,
    desc: "More, inclusive",
  },
};

const PROMPT_PRESETS = [
  {
    key: "vehicles",
    label: "Vehicles",
    prompt: "car . truck . bus . motorcycle",
    // icon: "🚗",
  },
  {
    key: "people",
    label: "People",
    prompt: "person . pedestrian",
    // icon: "🚶"
  },
  {
    key: "traffic",
    label: "Traffic",
    prompt: "traffic light . stop sign",
    // icon: "🚦",
  },
];

const CLASS_COLORS = {
  car: "#E53E3E",
  person: "#38A169",
  bicycle: "#3182CE",
  truck: "#D69E2E",
  dog: "#805AD5",
  cat: "#D53F8C",
  bus: "#319795",
  motorcycle: "#DD6B20",
  pedestrian: "#38A169",
  "traffic light": "#9F7AEA",
  "stop sign": "#F56565",
};

const REVIEW_STATUS = {
  auto: {
    label: "Auto",
    color: "#718096",
    // icon: "🤖"
  },
  approved: { label: "Approved", color: "#38A169", icon: "✓" },
  flagged: { label: "Flagged", color: "#D69E2E", icon: "⚠" },
};

// Canvas dimensions
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 500;

// ===== UTILITY FUNCTIONS =====

const imageToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

const getScaledBoundingBox = (region, imageWidth, imageHeight) => {
  if (!imageWidth || !imageHeight) {
    return { x: region.x, y: region.y, w: region.w, h: region.h };
  }

  const imageAspect = imageWidth / imageHeight;
  const canvasAspect = CANVAS_WIDTH / CANVAS_HEIGHT;

  let scale, offsetX, offsetY;

  if (imageAspect > canvasAspect) {
    scale = CANVAS_WIDTH / imageWidth;
    offsetX = 0;
    offsetY = (CANVAS_HEIGHT - CANVAS_WIDTH / imageAspect) / 2;
  } else {
    scale = CANVAS_HEIGHT / imageHeight;
    offsetX = (CANVAS_WIDTH - CANVAS_HEIGHT * imageAspect) / 2;
    offsetY = 0;
  }

  return {
    x: region.x * scale + offsetX,
    y: region.y * scale + offsetY,
    w: region.w * scale,
    h: region.h * scale,
  };
};

const checkModelHealth = async (baseUrl, healthEndpoint) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${baseUrl}${healthEndpoint}`, {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return { status: "connected", data };
    }
    return { status: "error", error: `Server returned ${response.status}` };
  } catch (error) {
    if (error.name === "AbortError") {
      return { status: "timeout", error: "Connection timeout" };
    }
    if (error.message.includes("Failed to fetch")) {
      return { status: "cors", error: "CORS error or server unreachable" };
    }
    return { status: "disconnected", error: error.message };
  }
};

const runDetection = async (baseUrl, endpoint, requestBody) => {
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }

    return response.json();
  } catch (error) {
    if (error.message.includes("Failed to fetch")) {
      throw new Error(
        "Network error: Check server is running and CORS enabled",
      );
    }
    throw error;
  }
};

// ===== EXPORT FUNCTIONS =====

const exportToYOLO = (images, regions) => {
  const allLabels = [
    ...new Set(
      Object.values(regions)
        .flat()
        .map((r) => r.label),
    ),
  ];
  const classToIdx = {};
  allLabels.forEach((name, idx) => {
    classToIdx[name] = idx;
  });

  const files = [];

  images.forEach((img) => {
    const imgRegions = regions[img.id] || [];
    const lines = imgRegions.map((r) => {
      const classIdx = classToIdx[r.label] ?? 0;
      const xCenter = (r.x + r.w / 2) / img.width;
      const yCenter = (r.y + r.h / 2) / img.height;
      const width = r.w / img.width;
      const height = r.h / img.height;
      return `${classIdx} ${xCenter.toFixed(6)} ${yCenter.toFixed(
        6,
      )} ${width.toFixed(6)} ${height.toFixed(6)}`;
    });

    files.push({
      filename: img.name.replace(/\.[^.]+$/, ".txt"),
      content: lines.join("\n"),
    });
  });

  files.push({ filename: "classes.txt", content: allLabels.join("\n") });
  return files;
};

const exportToCOCO = (images, regions) => {
  const categories = [];
  const categoryMap = {};
  const annotations = [];
  let annotationId = 1;

  const cocoImages = images.map((img, idx) => {
    (regions[img.id] || []).forEach((r) => {
      if (!categoryMap[r.label]) {
        const catId = categories.length + 1;
        categoryMap[r.label] = catId;
        categories.push({ id: catId, name: r.label, supercategory: "object" });
      }
      annotations.push({
        id: annotationId++,
        image_id: idx + 1,
        category_id: categoryMap[r.label],
        bbox: [
          Math.round(r.x),
          Math.round(r.y),
          Math.round(r.w),
          Math.round(r.h),
        ],
        area: Math.round(r.w * r.h),
        iscrowd: 0,
        score: r.confidence,
      });
    });
    return {
      id: idx + 1,
      file_name: img.name,
      width: img.width,
      height: img.height,
    };
  });

  return {
    info: {
      description: "AutoLabel Pro Export",
      version: "1.0",
      year: new Date().getFullYear(),
    },
    images: cocoImages,
    annotations,
    categories,
  };
};

const exportToVOC = (images, regions) => {
  return images.map((img) => {
    const objects = (regions[img.id] || [])
      .map(
        (r) => `
    <object>
        <name>${r.label}</name>
        <bndbox>
            <xmin>${Math.round(r.x)}</xmin>
            <ymin>${Math.round(r.y)}</ymin>
            <xmax>${Math.round(r.x + r.w)}</xmax>
            <ymax>${Math.round(r.y + r.h)}</ymax>
        </bndbox>
        <confidence>${r.confidence.toFixed(3)}</confidence>
    </object>`,
      )
      .join("");

    return {
      filename: img.name.replace(/\.[^.]+$/, ".xml"),
      content: `<?xml version="1.0"?>
<annotation>
    <filename>${img.name}</filename>
    <size><width>${img.width}</width><height>${img.height}</height><depth>3</depth></size>${objects}
</annotation>`,
    };
  });
};

// ===== COMPONENTS =====

const Toast = ({ message, type, onClose }) => {
  useEffect(() => {
    const t = setTimeout(onClose, 4000);
    return () => clearTimeout(t);
  }, [onClose]);
  const bg = {
    success: "#38A169",
    warning: "#D69E2E",
    error: "#E53E3E",
    info: "#3182CE",
  }[type];
  return (
    <div
      style={{
        position: "fixed",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        padding: "12px 20px",
        background: bg,
        color: "#FFF",
        borderRadius: 8,
        fontSize: 13,
        boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
        display: "flex",
        alignItems: "center",
        gap: 10,
        zIndex: 1000,
      }}
    >
      <span>{message}</span>
      <button
        onClick={onClose}
        style={{
          background: "none",
          border: "none",
          color: "#FFF",
          cursor: "pointer",
          fontSize: 16,
        }}
      >
        ×
      </button>
    </div>
  );
};

const ExportModal = ({ isOpen, onClose, onExport, totalLabels }) => {
  const [format, setFormat] = useState("yolo");
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: "#FFF",
          borderRadius: 12,
          padding: 24,
          width: 420,
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ margin: "0 0 16px", fontSize: 18 }}>
          Export {totalLabels} Labels
        </h3>
        <div style={{ marginBottom: 20 }}>
          {Object.entries(EXPORT_FORMATS).map(([key, fmt]) => (
            <label
              key={key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "12px 16px",
                marginBottom: 8,
                border: `2px solid ${format === key ? "#3182CE" : "#E2E8F0"}`,
                borderRadius: 8,
                cursor: "pointer",
                background: format === key ? "#EBF8FF" : "#FFF",
              }}
            >
              <input
                type="radio"
                checked={format === key}
                onChange={() => setFormat(key)}
              />
              <div>
                <div style={{ fontWeight: 600 }}>{fmt.name}</div>
                <div style={{ fontSize: 11, color: "#718096" }}>
                  {fmt.description}
                </div>
              </div>
            </label>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
          <button
            onClick={onClose}
            style={{
              padding: "10px 20px",
              background: "#FFF",
              border: "1px solid #E2E8F0",
              borderRadius: 8,
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => onExport(format)}
            style={{
              padding: "10px 24px",
              background: "#3182CE",
              color: "#FFF",
              border: "none",
              borderRadius: 8,
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Export
          </button>
        </div>
      </div>
    </div>
  );
};

// ===== MAIN COMPONENT =====
export default function AutoLabelProV6() {
  const fileInputRef = useRef(null);

  // Server config - separate URL per model type
  const [serverUrls, setServerUrls] = useState({
    "grounding-dino": "http://localhost:8000",
    "yolo-v8": "http://localhost:8001",
    "yolo-v11": "http://localhost:8001",
  });
  const [showServerConfig, setShowServerConfig] = useState(false);

  // Model state
  const [selectedModel, setSelectedModel] = useState("grounding-dino");
  const [modelStatus, setModelStatus] = useState("checking");
  const [thresholds, setThresholds] = useState({});
  const [sensitivityPreset, setSensitivityPreset] = useState("balanced");
  const [showThresholds, setShowThresholds] = useState(false);

  // UI state
  const [currentStep, setCurrentStep] = useState(1);
  const [prompt, setPrompt] = useState("car . person . bicycle");
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [showConfidence, setShowConfidence] = useState(true);
  const [toast, setToast] = useState(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Image state
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [regions, setRegions] = useState({});

  // Derived
  const currentModelConfig = MODEL_CONFIGS[selectedModel];
  const currentServerUrl = serverUrls[selectedModel];
  const currentImage = images[currentImageIndex] || null;
  const currentRegions = currentImage ? regions[currentImage.id] || [] : [];
  const totalRegionsAllImages = Object.values(regions).flat().length;

  const showToast = useCallback(
    (msg, type = "success") => setToast({ message: msg, type }),
    [],
  );

  // Initialize thresholds when model changes
  useEffect(() => {
    if (currentModelConfig?.thresholds) {
      const defaults = {};
      Object.entries(currentModelConfig.thresholds).forEach(([k, v]) => {
        defaults[k] = v.default;
      });
      setThresholds(defaults);
    }
  }, [selectedModel]);

  // Check model health
  useEffect(() => {
    const check = async () => {
      setModelStatus("checking");
      const result = await checkModelHealth(
        currentServerUrl,
        currentModelConfig?.healthEndpoint || "/health",
      );
      setModelStatus(
        result.status === "connected" ? "connected" : "disconnected",
      );
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, [selectedModel, currentServerUrl]);

  // Apply sensitivity preset
  const applySensitivityPreset = (key) => {
    setSensitivityPreset(key);
    const preset = SENSITIVITY_PRESETS[key];
    if (currentModelConfig?.thresholds) {
      const newT = {};
      Object.entries(currentModelConfig.thresholds).forEach(([k, v]) => {
        newT[k] = Math.max(
          v.min,
          Math.min(v.max, v.default * preset.multiplier),
        );
      });
      setThresholds(newT);
    }
  };

  // Handlers
  const handleImportClick = () => fileInputRef.current?.click();

  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    const newImages = await Promise.all(
      files.map(async (file, idx) => {
        const url = URL.createObjectURL(file);
        const base64 = await imageToBase64(file);
        const img = new Image();
        await new Promise((r) => {
          img.onload = r;
          img.src = url;
        });
        return {
          id: `img_${Date.now()}_${idx}`,
          name: file.name,
          url,
          base64,
          width: img.width,
          height: img.height,
        };
      }),
    );

    setImages((prev) => [...prev, ...newImages]);
    if (currentStep === 1) setCurrentStep(2);
    showToast(`${files.length} image(s) imported`, "success");
    e.target.value = "";
  };

  const handleDetect = async () => {
    if (!currentImage) {
      showToast("Import images first", "warning");
      return;
    }
    if (modelStatus !== "connected") {
      showToast(
        `${currentModelConfig.name} not connected. Check server at ${currentServerUrl}`,
        "error",
      );
      return;
    }
    if (currentModelConfig.requiresPrompt && !prompt.trim()) {
      showToast("Enter detection prompt", "warning");
      return;
    }

    setIsDetecting(true);
    try {
      let requestBody;
      if (selectedModel === "grounding-dino") {
        requestBody = {
          image: currentImage.base64,
          prompt,
          box_threshold: thresholds.box || 0.35,
          text_threshold: thresholds.text || 0.25,
          nms_threshold: thresholds.nms || 0.5,
        };
      } else {
        requestBody = {
          image: currentImage.base64,
          confidence_threshold: thresholds.confidence || 0.25,
          iou_threshold: thresholds.iou || 0.45,
        };
      }

      const result = await runDetection(
        currentServerUrl,
        currentModelConfig.endpoint,
        requestBody,
      );
      const detections = (result.detections || []).map((det, idx) => ({
        id: Date.now() + idx,
        label: det.label || det.class || "object",
        confidence: det.confidence || det.score || 0,
        status: "auto",
        x: det.bbox?.[0] || 0,
        y: det.bbox?.[1] || 0,
        w: det.bbox?.[2] || 100,
        h: det.bbox?.[3] || 100,
      }));

      setRegions((prev) => ({ ...prev, [currentImage.id]: detections }));
      setCurrentStep(3);
      showToast(`${detections.length} objects detected`, "success");
    } catch (err) {
      showToast(`Detection failed: ${err.message}`, "error");
    } finally {
      setIsDetecting(false);
    }
  };

  const handleExport = (format) => {
    setShowExportModal(false);
    if (totalRegionsAllImages === 0) {
      showToast("No labels to export", "warning");
      return;
    }

    const ts = Date.now();
    try {
      if (format === "yolo") {
        const files = exportToYOLO(images, regions);
        const content = files
          .map((f) => `=== ${f.filename} ===\n${f.content}`)
          .join("\n\n");
        downloadFile(content, `yolo_labels_${ts}.txt`, "text/plain");
      } else if (format === "coco") {
        const data = exportToCOCO(images, regions);
        downloadFile(
          JSON.stringify(data, null, 2),
          `coco_${ts}.json`,
          "application/json",
        );
      } else if (format === "voc") {
        const files = exportToVOC(images, regions);
        const content = files
          .map((f) => `=== ${f.filename} ===\n${f.content}`)
          .join("\n\n");
        downloadFile(content, `voc_${ts}.xml`, "text/plain");
      } else {
        const data = {
          images: images.map((img) => ({
            filename: img.name,
            width: img.width,
            height: img.height,
            annotations: (regions[img.id] || []).map((r) => ({
              label: r.label,
              confidence: r.confidence,
              status: r.status,
              bbox: { x: r.x, y: r.y, w: r.w, h: r.h },
            })),
          })),
        };
        downloadFile(
          JSON.stringify(data, null, 2),
          `annotations_${ts}.json`,
          "application/json",
        );
      }
      setCurrentStep(4);
      showToast(
        `Exported ${totalRegionsAllImages} labels in ${format.toUpperCase()} format`,
        "success",
      );
    } catch (err) {
      showToast(`Export failed: ${err.message}`, "error");
    }
  };

  const downloadFile = (content, filename, type) => {
    const blob = new Blob([content], { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // Region actions
  const updateRegions = (fn) =>
    currentImage &&
    setRegions((prev) => ({
      ...prev,
      [currentImage.id]: fn(prev[currentImage.id] || []),
    }));

  const handleApprove = () => {
    if (selectedRegions.length) {
      updateRegions((rs) =>
        rs.map((r) =>
          selectedRegions.includes(r.id) ? { ...r, status: "approved" } : r,
        ),
      );
      showToast(`${selectedRegions.length} approved`, "success");
      setSelectedRegions([]);
    }
  };
  const handleFlag = () => {
    if (selectedRegions.length) {
      updateRegions((rs) =>
        rs.map((r) =>
          selectedRegions.includes(r.id) ? { ...r, status: "flagged" } : r,
        ),
      );
      showToast(`${selectedRegions.length} flagged`, "warning");
    }
  };
  const handleDelete = () => {
    if (selectedRegions.length) {
      const n = selectedRegions.length;
      updateRegions((rs) => rs.filter((r) => !selectedRegions.includes(r.id)));
      showToast(`${n} deleted`, "info");
      setSelectedRegions([]);
    }
  };

  const handleSelectRegion = (id, e) => {
    if (e?.metaKey || e?.ctrlKey)
      setSelectedRegions((prev) =>
        prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
      );
    else setSelectedRegions([id]);
  };

  // Zoom
  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.25, 4));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.25, 0.5));
  const handleZoomReset = () => setZoom(1);

  // Keyboard
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")
        return;
      if (e.key === "ArrowLeft" && currentImageIndex > 0) {
        setCurrentImageIndex((i) => i - 1);
        setSelectedRegions([]);
      }
      if (e.key === "ArrowRight" && currentImageIndex < images.length - 1) {
        setCurrentImageIndex((i) => i + 1);
        setSelectedRegions([]);
      }
      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        selectedRegions.length
      ) {
        e.preventDefault();
        handleDelete();
      }
      if (e.key === "Escape") setSelectedRegions([]);
      if (e.key === "=" || e.key === "+") handleZoomIn();
      if (e.key === "-") handleZoomOut();
      if (e.key === "0") handleZoomReset();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentImageIndex, images.length, selectedRegions]);

  // Model groups
  const modelsByCategory = Object.values(MODEL_CONFIGS).reduce((acc, m) => {
    (acc[m.category] = acc[m.category] || []).push(m);
    return acc;
  }, {});

  // Styles
  const s = {
    container: {
      fontFamily: "'Inter', sans-serif",
      height: "97vh",
      display: "flex",
      flexDirection: "column",
      background: "#F7FAFC",
      overflow: "hidden",
    },
    header: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "1px 20px",
      background: "#FFF",
      borderBottom: "1px solid #E2E8F0",
    },
    stepper: { display: "flex", alignItems: "center", gap: 6 },
    step: (active, done) => ({
      display: "flex",
      alignItems: "center",
      gap: 6,
      padding: "6px 14px",
      borderRadius: 6,
      fontSize: 12,
      fontWeight: 500,
      background: active ? "#EBF8FF" : done ? "#F0FFF4" : "#F7FAFC",
      color: active ? "#2B6CB0" : done ? "#276749" : "#A0AEC0",
      border: active ? "1px solid #90CDF4" : "1px solid transparent",
      cursor: "pointer",
    }),
    stepNum: (active, done) => ({
      width: 20,
      height: 20,
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 10,
      fontWeight: 600,
      background: done ? "#38A169" : active ? "#3182CE" : "#E2E8F0",
      color: done || active ? "#FFF" : "#A0AEC0",
    }),
    controls: {
      padding: "10px 20px",
      background: "#FFF",
      borderBottom: "1px solid #E2E8F0",
      display: "flex",
      gap: 16,
      alignItems: "flex-start",
    },
    main: { flex: 1, display: "flex", overflow: "hidden" },
    leftPanel: {
      width: 180,
      background: "#FFF",
      borderRight: "1px solid #E2E8F0",
      display: "flex",
      flexDirection: "column",
    },
    rightPanel: {
      width: 280,
      background: "#FFF",
      borderLeft: "1px solid #E2E8F0",
      display: "flex",
      flexDirection: "column",
    },
    canvas: {
      flex: 1,
      background: "#1A202C",
      display: "flex",
      flexDirection: "column",
    },
    canvasToolbar: {
      padding: "6px 12px",
      background: "#2D3748",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    },
    canvasArea: {
      flex: 1,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      overflow: "hidden",
      padding: 12,
    },
    btn: (primary, disabled) => ({
      padding: "8px 16px",
      fontSize: 13,
      fontWeight: 600,
      borderRadius: 6,
      border: "none",
      cursor: disabled ? "not-allowed" : "pointer",
      background: disabled ? "#A0AEC0" : primary ? "#48BB78" : "#FFF",
      color: disabled ? "#FFF" : primary ? "#FFF" : "#4A5568",
      borderColor: primary ? "transparent" : "#E2E8F0",
      borderWidth: 1,
      borderStyle: "solid",
    }),
    sectionTitle: {
      fontSize: 11,
      fontWeight: 600,
      color: "#718096",
      textTransform: "uppercase",
      marginBottom: 8,
    },
  };

  return (
    <div style={s.container}>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
      <ExportModal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        onExport={handleExport}
        totalLabels={totalRegionsAllImages}
      />

      {/* HEADER */}
      <header style={s.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 42,
              height: 42,
              // background: "linear-gradient(135deg, #4299E1, #2B6CB0)",

              borderRadius: 3,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              // color: "#FFF",
              color: "#0c0c0c",
              fontWeight: 700,
              fontSize: 20,
            }}
          >
            {/* AL */}
            逢甲
          </div>
          <span style={{ fontSize: 16, fontWeight: 600 }}>
            {/* AutoLabel <span style={{ color: "#3182CE" }}>Pro</span> */}
            <p>自動駛駛研究團隊</p>
          </span>
        </div>

        {/* STEPPER */}
        <div style={s.stepper}>
          {["Import", "Detect", "Review", "Export"].map((label, i) => (
            <React.Fragment key={label}>
              <div
                style={s.step(currentStep === i + 1, currentStep > i + 1)}
                onClick={() => {
                  if (i + 1 <= currentStep || i === 0) setCurrentStep(i + 1);
                }}
              >
                <span
                  style={s.stepNum(currentStep === i + 1, currentStep > i + 1)}
                >
                  {currentStep > i + 1 ? "✓" : i + 1}
                </span>
                {label}
              </div>
              {i < 3 && <span style={{ color: "#CBD5E0" }}>›</span>}
            </React.Fragment>
          ))}
        </div>

        <button
          onClick={handleImportClick}
          style={{
            padding: "8px 16px",
            fontSize: 13,
            background: "#FFF",
            border: "1px solid #E2E8F0",
            borderRadius: 6,
            cursor: "pointer",
            fontWeight: 500,
          }}
        >
          📁 Import Images
        </button>
      </header>

      {/* CONTROLS */}
      <div style={s.controls}>
        {/* Model Selector */}
        <div style={{ minWidth: 180 }}>
          <div style={s.sectionTitle}>Detection Model</div>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 10px",
              fontSize: 13,
              border: "1px solid #E2E8F0",
              borderRadius: 6,
            }}
          >
            {Object.entries(modelsByCategory).map(([cat, models]) => (
              <optgroup key={cat} label={cat}>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              marginTop: 6,
              fontSize: 11,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background:
                  modelStatus === "connected"
                    ? "#38A169"
                    : modelStatus === "checking"
                      ? "#D69E2E"
                      : "#E53E3E",
              }}
            />
            <span
              style={{
                color: modelStatus === "connected" ? "#38A169" : "#718096",
              }}
            >
              {modelStatus === "connected"
                ? "Connected"
                : modelStatus === "checking"
                  ? "Checking..."
                  : "Disconnected"}
            </span>
          </div>
          <button
            onClick={() => setShowServerConfig(!showServerConfig)}
            style={{
              marginTop: 4,
              fontSize: 10,
              color: "#718096",
              background: "none",
              border: "none",
              cursor: "pointer",
            }}
          >
            ⚙️ {showServerConfig ? "Hide" : "Server URL"}
          </button>
          {showServerConfig && (
            <input
              type="text"
              value={currentServerUrl}
              onChange={(e) =>
                setServerUrls((prev) => ({
                  ...prev,
                  [selectedModel]: e.target.value,
                }))
              }
              placeholder="http://localhost:8000"
              style={{
                width: "100%",
                padding: "6px 8px",
                fontSize: 11,
                border: "1px solid #E2E8F0",
                borderRadius: 4,
                marginTop: 4,
              }}
            />
          )}
        </div>

        {/* Prompt Panel - LARGER */}
        {currentModelConfig?.requiresPrompt && (
          <div style={{ flex: 1, maxWidth: 350 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span style={s.sectionTitle}>What to detect?</span>
              <span style={{ fontSize: 10, color: "#A0AEC0" }}>
                Separate with periods (.)
              </span>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="car . person . bicycle"
              style={{
                width: "100%",
                minHeight: 60,
                padding: "10px 12px",
                fontSize: 13,
                border: "1px solid #E2E8F0",
                borderRadius: 6,
                resize: "none",
                fontFamily: "inherit",
              }}
            />
            <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
              {PROMPT_PRESETS.map((p) => (
                <button
                  key={p.key}
                  onClick={() => {
                    setSelectedPreset(p.key);
                    setPrompt(p.prompt);
                  }}
                  style={{
                    padding: "5px 10px",
                    fontSize: 11,
                    border: `1px solid ${
                      selectedPreset === p.key ? "#3182CE" : "#E2E8F0"
                    }`,
                    borderRadius: 4,
                    background: selectedPreset === p.key ? "#EBF8FF" : "#FFF",
                    cursor: "pointer",
                  }}
                >
                  {p.icon} {p.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Detect Button */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <button
            onClick={handleDetect}
            disabled={
              isDetecting || !currentImage || modelStatus !== "connected"
            }
            style={s.btn(
              true,
              isDetecting || !currentImage || modelStatus !== "connected",
            )}
          >
            {isDetecting ? " Detecting..." : " Find Objects"}
          </button>
        </div>
      </div>

      {/* MAIN */}
      <div style={s.main}>
        {/* LEFT PANEL - Images */}
        <div style={s.leftPanel}>
          <div
            style={{
              padding: "10px 12px",
              borderBottom: "1px solid #E2E8F0",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 12, fontWeight: 600 }}>
              Images ({images.length})
            </span>
            <button
              onClick={handleImportClick}
              style={{
                padding: "3px 8px",
                fontSize: 10,
                background: "#EBF8FF",
                color: "#2B6CB0",
                border: "1px solid #90CDF4",
                borderRadius: 4,
                cursor: "pointer",
              }}
            >
              + Add
            </button>
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 8 }}>
            {images.length === 0 ? (
              <div
                style={{ textAlign: "center", padding: 24, color: "#A0AEC0" }}
              >
                <div style={{ fontSize: 28, marginBottom: 8 }}>📁</div>
                <div style={{ fontSize: 11 }}>No images</div>
              </div>
            ) : (
              images.map((img, i) => (
                <div
                  key={img.id}
                  onClick={() => {
                    setCurrentImageIndex(i);
                    setSelectedRegions([]);
                    setZoom(1);
                  }}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px",
                    marginBottom: 4,
                    borderRadius: 6,
                    cursor: "pointer",
                    background: i === currentImageIndex ? "#EBF8FF" : "#FFF",
                    border: `1px solid ${
                      i === currentImageIndex ? "#90CDF4" : "#E2E8F0"
                    }`,
                  }}
                >
                  <div
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 4,
                      overflow: "hidden",
                      background: "#E2E8F0",
                      flexShrink: 0,
                    }}
                  >
                    <img
                      src={img.url}
                      alt=""
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                      }}
                    />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 10,
                        fontWeight: 500,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {img.name}
                    </div>
                    <div style={{ fontSize: 9, color: "#A0AEC0" }}>
                      {(regions[img.id] || []).length} objects
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* CENTER - Canvas */}
        <div style={s.canvas}>
          <div style={s.canvasToolbar}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button
                onClick={() =>
                  currentImageIndex > 0 && setCurrentImageIndex((i) => i - 1)
                }
                style={{
                  padding: "4px 8px",
                  background: "#4A5568",
                  color: "#FFF",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                }}
              >
                ←
              </button>
              <span style={{ fontSize: 12, color: "#E2E8F0" }}>
                {currentImage?.name || "No image"}
              </span>
              <span style={{ fontSize: 11, color: "#718096" }}>
                {currentImageIndex + 1}/{Math.max(1, images.length)}
              </span>
              <button
                onClick={() =>
                  currentImageIndex < images.length - 1 &&
                  setCurrentImageIndex((i) => i + 1)
                }
                style={{
                  padding: "4px 8px",
                  background: "#4A5568",
                  color: "#FFF",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                }}
              >
                →
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button
                onClick={handleZoomOut}
                style={{
                  padding: "4px 8px",
                  background: "#4A5568",
                  color: "#FFF",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                }}
              >
                −
              </button>
              <span
                style={{
                  fontSize: 11,
                  color: "#E2E8F0",
                  minWidth: 45,
                  textAlign: "center",
                }}
              >
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={handleZoomIn}
                style={{
                  padding: "4px 8px",
                  background: "#4A5568",
                  color: "#FFF",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                }}
              >
                +
              </button>
              <button
                onClick={handleZoomReset}
                style={{
                  padding: "4px 8px",
                  fontSize: 10,
                  background: "#4A5568",
                  color: "#FFF",
                  border: "none",
                  borderRadius: 4,
                  cursor: "pointer",
                }}
              >
                Reset
              </button>
            </div>
            <span style={{ fontSize: 11, color: "#718096" }}>
              {currentRegions.length} objects
            </span>
          </div>

          <div style={s.canvasArea}>
            {currentImage ? (
              <div
                style={{
                  width: CANVAS_WIDTH,
                  height: CANVAS_HEIGHT,
                  position: "relative",
                  overflow: "hidden",
                  borderRadius: 8,
                  transform: `scale(${zoom})`,
                  transformOrigin: "center",
                  transition: "transform 0.15s",
                }}
              >
                <img
                  src={currentImage.url}
                  alt=""
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                  }}
                />
                {currentRegions.map((region) => {
                  const isSelected = selectedRegions.includes(region.id);
                  const isHovered = hoveredRegion === region.id;
                  const scaled = getScaledBoundingBox(
                    region,
                    currentImage.width,
                    currentImage.height,
                  );
                  const color = CLASS_COLORS[region.label] || "#3182CE";
                  return (
                    <div
                      key={region.id}
                      onClick={(e) => handleSelectRegion(region.id, e)}
                      onMouseEnter={() => setHoveredRegion(region.id)}
                      onMouseLeave={() => setHoveredRegion(null)}
                      style={{
                        position: "absolute",
                        left: scaled.x,
                        top: scaled.y,
                        width: scaled.w,
                        height: scaled.h,
                        border: `${isSelected ? 4 : 3}px solid ${color}`,
                        background: isSelected
                          ? `${color}30`
                          : isHovered
                            ? `${color}20`
                            : "transparent",
                        borderRadius: 4,
                        cursor: "pointer",
                        boxSizing: "border-box",
                      }}
                    >
                      {(isSelected || isHovered) && (
                        <div
                          style={{
                            position: "absolute",
                            top: -24,
                            left: -2,
                            padding: "3px 8px",
                            background: color,
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            color: "#FFF",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {region.label}{" "}
                          {showConfidence && (
                            <span style={{ opacity: 0.8 }}>
                              {(region.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ textAlign: "center", color: "#718096" }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>{/* 📷 */}</div>
                <div style={{ marginBottom: 12 }}>No image selected</div>
                <button
                  onClick={handleImportClick}
                  style={{
                    padding: "10px 20px",
                    background: "#3182CE",
                    color: "#FFF",
                    border: "none",
                    borderRadius: 8,
                    cursor: "pointer",
                  }}
                >
                  Import Images
                </button>
              </div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL - Labeled Objects + Controls */}
        <div style={s.rightPanel}>
          {/* Sensitivity & Thresholds - MOVED HERE */}
          <div style={{ padding: "12px", borderBottom: "1px solid #E2E8F0" }}>
            <div style={s.sectionTitle}>Sensitivity</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 10 }}>
              {Object.entries(SENSITIVITY_PRESETS).map(([key, p]) => (
                <button
                  key={key}
                  onClick={() => applySensitivityPreset(key)}
                  style={{
                    flex: 1,
                    padding: "6px",
                    fontSize: 10,
                    textAlign: "center",
                    border: `1px solid ${
                      sensitivityPreset === key ? "#3182CE" : "#E2E8F0"
                    }`,
                    borderRadius: 4,
                    background: sensitivityPreset === key ? "#EBF8FF" : "#FFF",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ fontSize: 14 }}>{p.icon}</div>
                  <div style={{ fontWeight: 600 }}>{p.label}</div>
                </button>
              ))}
            </div>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 6,
              }}
            >
              <span style={s.sectionTitle}>Thresholds</span>
              <button
                onClick={() => setShowThresholds(!showThresholds)}
                style={{
                  fontSize: 10,
                  color: "#718096",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                {showThresholds ? "Hide" : "Show"}
              </button>
            </div>
            {showThresholds && currentModelConfig?.thresholds && (
              <div style={{ marginBottom: 8 }}>
                {Object.entries(currentModelConfig.thresholds).map(([k, v]) => (
                  <div
                    key={k}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      marginBottom: 6,
                    }}
                  >
                    <span
                      style={{ fontSize: 10, color: "#718096", minWidth: 60 }}
                    >
                      {v.label}
                    </span>
                    <input
                      type="range"
                      min={v.min}
                      max={v.max}
                      step={0.05}
                      value={thresholds[k] || v.default}
                      onChange={(e) => {
                        setThresholds((p) => ({
                          ...p,
                          [k]: parseFloat(e.target.value),
                        }));
                        setSensitivityPreset(null);
                      }}
                      style={{ flex: 1, cursor: "pointer" }}
                    />
                    <span
                      style={{ fontSize: 10, fontWeight: 600, minWidth: 30 }}
                    >
                      {((thresholds[k] || v.default) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 11,
                color: "#718096",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={showConfidence}
                onChange={(e) => setShowConfidence(e.target.checked)}
              />
              Show confidence
            </label>
          </div>

          {/* Labeled Objects */}
          <div
            style={{
              padding: "10px 12px",
              borderBottom: "1px solid #E2E8F0",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 12, fontWeight: 600 }}>
              Labeled Objects
            </span>
            <span
              style={{
                padding: "2px 8px",
                background: "#EDF2F7",
                borderRadius: 10,
                fontSize: 11,
              }}
            >
              {currentRegions.length}
            </span>
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 8 }}>
            {currentRegions.length === 0 ? (
              <div
                style={{
                  textAlign: "center",
                  padding: 24,
                  color: "#A0AEC0",
                  fontSize: 11,
                }}
              >
                No objects detected
              </div>
            ) : (
              currentRegions.map((region) => {
                const isSelected = selectedRegions.includes(region.id);
                const color = CLASS_COLORS[region.label] || "#3182CE";
                const status =
                  REVIEW_STATUS[region.status] || REVIEW_STATUS.auto;
                return (
                  <div
                    key={region.id}
                    onClick={(e) => handleSelectRegion(region.id, e)}
                    onMouseEnter={() => setHoveredRegion(region.id)}
                    onMouseLeave={() => setHoveredRegion(null)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "8px",
                      marginBottom: 4,
                      borderRadius: 6,
                      cursor: "pointer",
                      background: isSelected ? "#EBF8FF" : "#FFF",
                      border: `1px solid ${isSelected ? "#63B3ED" : "#E2E8F0"}`,
                    }}
                  >
                    <div
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: color,
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ flex: 1, fontSize: 11, fontWeight: 500 }}>
                      {region.label}
                    </span>
                    <span style={{ fontSize: 10, color: "#718096" }}>
                      {(region.confidence * 100).toFixed(0)}%
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 8,
                        background: `${status.color}20`,
                        color: status.color,
                      }}
                    >
                      {status.icon}
                    </span>
                  </div>
                );
              })
            )}
          </div>

          {/* Quick Actions - RESTORED */}
          <div
            style={{
              padding: "12px",
              borderTop: "1px solid #E2E8F0",
              background: "#F7FAFC",
            }}
          >
            <div style={{ fontSize: 11, color: "#718096", marginBottom: 8 }}>
              {selectedRegions.length > 0
                ? `${selectedRegions.length} selected`
                : "Select objects"}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={handleApprove}
                disabled={!selectedRegions.length}
                style={{
                  flex: 1,
                  padding: "6px",
                  fontSize: 11,
                  background: selectedRegions.length ? "#38A169" : "#E2E8F0",
                  color: selectedRegions.length ? "#FFF" : "#A0AEC0",
                  border: "none",
                  borderRadius: 4,
                  cursor: selectedRegions.length ? "pointer" : "not-allowed",
                }}
              >
                ✓ Approve
              </button>
              <button
                onClick={handleFlag}
                disabled={!selectedRegions.length}
                style={{
                  flex: 1,
                  padding: "6px",
                  fontSize: 11,
                  background: selectedRegions.length ? "#FFF" : "#E2E8F0",
                  color: selectedRegions.length ? "#D69E2E" : "#A0AEC0",
                  border: "1px solid",
                  borderColor: selectedRegions.length ? "#D69E2E" : "#E2E8F0",
                  borderRadius: 4,
                  cursor: selectedRegions.length ? "pointer" : "not-allowed",
                }}
              >
                ⚠ Flag
              </button>
              <button
                onClick={handleDelete}
                disabled={!selectedRegions.length}
                style={{
                  flex: 1,
                  padding: "6px",
                  fontSize: 11,
                  background: selectedRegions.length ? "#FFF" : "#E2E8F0",
                  color: selectedRegions.length ? "#E53E3E" : "#A0AEC0",
                  border: "1px solid",
                  borderColor: selectedRegions.length ? "#E53E3E" : "#E2E8F0",
                  borderRadius: 4,
                  cursor: selectedRegions.length ? "pointer" : "not-allowed",
                }}
              >
                🗑 Delete
              </button>
            </div>
          </div>

          {/* Export Button */}
          <div style={{ padding: "12px", borderTop: "1px solid #E2E8F0" }}>
            <button
              onClick={() => setShowExportModal(true)}
              disabled={totalRegionsAllImages === 0}
              style={{
                width: "100%",
                padding: "10px",
                fontSize: 13,
                fontWeight: 600,
                background: totalRegionsAllImages === 0 ? "#E2E8F0" : "#3182CE",
                color: totalRegionsAllImages === 0 ? "#A0AEC0" : "#FFF",
                border: "none",
                borderRadius: 6,
                cursor: totalRegionsAllImages === 0 ? "not-allowed" : "pointer",
              }}
            >
              Export ({totalRegionsAllImages} labels)
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "6px 20px",
          background: "#FFF",
          borderTop: "1px solid #E2E8F0",
          fontSize: 10,
          color: "#718096",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", gap: 16 }}>
          <span>← → Navigate</span>
          <span>+ − Zoom</span>
          <span>Del Delete</span>
        </div>
        <span>
          {currentModelConfig?.name} @ {currentServerUrl}
        </span>
      </div>
    </div>
  );
}
