import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";

/**
 * AutoLabel Pro v7.3 — Professional Edition
 *
 * New in v7.3:
 * - Collapsible detection controls (maximize canvas space)
 * - Image status indicators (gray=todo, yellow=detected, green=approved)
 * - Progress bar for batch workflows
 * - Shift+Click range selection
 * - Right-click context menu for bulk actions
 * - Keyboard shortcut tooltips on buttons
 * - Confidence-based border styling (dashed for low confidence)
 *
 * Previous features:
 * - Grouped labeled objects with collapsible accordions
 * - Filter chips with canvas dimming
 * - Clear/detect confirmation modals
 * - Label display modes (Always/Hover/Hidden)
 */

// ===== MODEL CONFIGURATIONS =====
const MODEL_CONFIGS = {
  "yolo-v8": {
    id: "yolo-v8", name: "YOLOv8", category: "Closed-Set Detection", requiresPrompt: false,
    endpoint: "/yolo", healthEndpoint: "/health", defaultPort: 8001,
    thresholds: { confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 }, iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 } },
  },
  "yolo-v11": {
    id: "yolo-v11", name: "YOLOv11", category: "Closed-Set Detection", requiresPrompt: false,
    endpoint: "/yolo11", healthEndpoint: "/health", defaultPort: 8001,
    thresholds: { confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 }, iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 } },
  },
  "grounding-dino": {
    id: "grounding-dino", name: "VLM Model", category: "Open-World Detection", requiresPrompt: true,
    endpoint: "/grounding-dino", healthEndpoint: "/health", defaultPort: 8000,
    thresholds: { box: { label: "Box Threshold", default: 0.35, min: 0.1, max: 0.9 }, text: { label: "Text Threshold", default: 0.25, min: 0.1, max: 0.9 }, nms: { label: "NMS Threshold", default: 0.5, min: 0.1, max: 0.9 } },
  },
  "coming-soon": {
    id: "coming-soon", name: "Coming Soon", category: "Open-World Detection", requiresPrompt: false, disabled: true,
    endpoint: "/yolo11", healthEndpoint: "/health", defaultPort: 8001,
    thresholds: { confidence: { label: "Confidence", default: 0.25, min: 0.05, max: 0.95 }, iou: { label: "IoU (NMS)", default: 0.45, min: 0.1, max: 0.9 } },
  },
};

const EXPORT_FORMATS = {
  yolo: { name: "YOLO", ext: "txt", description: "class x_center y_center width height (normalized)" },
  coco: { name: "COCO JSON", ext: "json", description: "Standard COCO dataset format" },
  voc: { name: "Pascal VOC", ext: "xml", description: "XML annotation format" },
  json: { name: "JSON", ext: "json", description: "Simple JSON format" },
};

const SENSITIVITY_PRESETS = {
  strict: { label: "Strict", multiplier: 1.4 },
  balanced: { label: "Balanced", multiplier: 1.0 },
  loose: { label: "Loose", multiplier: 0.7 },
};

const PROMPT_PRESETS = [
  { key: "vehicles", label: "Vehicles", prompt: "car . truck . bus . motorcycle" },
  { key: "people", label: "People", prompt: "person . pedestrian" },
  { key: "traffic", label: "Traffic", prompt: "traffic light . stop sign" },
];

const CLASS_COLORS = {
  car: "#E53E3E", person: "#38A169", bicycle: "#3182CE", truck: "#D69E2E",
  dog: "#805AD5", cat: "#D53F8C", bus: "#319795", motorcycle: "#DD6B20",
  pedestrian: "#38A169", "traffic light": "#9F7AEA", "stop sign": "#F56565",
};

const REVIEW_STATUS = {
  auto: { label: "Auto", color: "#718096" },
  approved: { label: "Approved", color: "#38A169", icon: "✓" },
  flagged: { label: "Flagged", color: "#D69E2E", icon: "⚠" },
  manual: { label: "Manual", color: "#805AD5", icon: "✎" },
};

// Image workflow status
const IMAGE_STATUS = {
  todo: { label: "To Do", color: "#A0AEC0", icon: "○" },
  detected: { label: "Detected", color: "#D69E2E", icon: "◐" },
  reviewed: { label: "Reviewed", color: "#38A169", icon: "●" },
};

const ANNOTATION_MODES = {
  select: { label: "Select", icon: "↖", cursor: "default", shortcut: "V" },
  draw: { label: "Draw", icon: "□", cursor: "crosshair", shortcut: "D" },
  edit: { label: "Edit", icon: "✎", cursor: "move", shortcut: "E" },
};

const LABEL_DISPLAY_MODES = {
  always: { label: "Always", desc: "Show all labels" },
  hover: { label: "On Hover", desc: "Show on hover only" },
  hidden: { label: "Hidden", desc: "Hide all labels" },
};

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 500;
const HANDLE_SIZE = 8;
const PRIMARY_COLOR = "#3182CE";
const PRIMARY_LIGHT = "#EBF8FF";
const PRIMARY_BORDER = "#90CDF4";
const LOW_CONFIDENCE_THRESHOLD = 0.5;

// ===== UTILITY FUNCTIONS =====
const imageToBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = reject;
  reader.readAsDataURL(file);
});

const getImageTransform = (imageWidth, imageHeight) => {
  if (!imageWidth || !imageHeight) return { scale: 1, offsetX: 0, offsetY: 0 };
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
  return { scale, offsetX, offsetY };
};

const getScaledBoundingBox = (region, imageWidth, imageHeight) => {
  const { scale, offsetX, offsetY } = getImageTransform(imageWidth, imageHeight);
  return { x: region.x * scale + offsetX, y: region.y * scale + offsetY, w: region.w * scale, h: region.h * scale };
};

const canvasToImageCoords = (canvasX, canvasY, imageWidth, imageHeight) => {
  const { scale, offsetX, offsetY } = getImageTransform(imageWidth, imageHeight);
  return { x: (canvasX - offsetX) / scale, y: (canvasY - offsetY) / scale };
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const getClassColor = (label) => {
  if (CLASS_COLORS[label]) return CLASS_COLORS[label];
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = label.charCodeAt(i) + ((hash << 5) - hash);
  return `hsl(${Math.abs(hash % 360)}, 65%, 50%)`;
};

const checkModelHealth = async (baseUrl, healthEndpoint) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const response = await fetch(`${baseUrl}${healthEndpoint}`, { method: "GET", signal: controller.signal });
    clearTimeout(timeoutId);
    if (response.ok) return { status: "connected", data: await response.json() };
    return { status: "error", error: `Server returned ${response.status}` };
  } catch (error) {
    if (error.name === "AbortError") return { status: "timeout", error: "Connection timeout" };
    if (error.message.includes("Failed to fetch")) return { status: "cors", error: "CORS error or server unreachable" };
    return { status: "disconnected", error: error.message };
  }
};

const runDetection = async (baseUrl, endpoint, requestBody) => {
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestBody) });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  } catch (error) {
    if (error.message.includes("Failed to fetch")) throw new Error("Network error: Check server is running and CORS enabled");
    throw error;
  }
};

// Get image workflow status
const getImageStatus = (imageRegions) => {
  if (!imageRegions || imageRegions.length === 0) return "todo";
  const allApproved = imageRegions.every(r => r.status === "approved");
  if (allApproved) return "reviewed";
  return "detected";
};

// ===== EXPORT FUNCTIONS =====
const exportToYOLO = (images, regions) => {
  const allLabels = [...new Set(Object.values(regions).flat().map((r) => r.label))];
  const classToIdx = {};
  allLabels.forEach((name, idx) => { classToIdx[name] = idx; });
  const files = images.map((img) => {
    const imgRegions = regions[img.id] || [];
    const lines = imgRegions.map((r) => {
      const classIdx = classToIdx[r.label] ?? 0;
      return `${classIdx} ${((r.x + r.w / 2) / img.width).toFixed(6)} ${((r.y + r.h / 2) / img.height).toFixed(6)} ${(r.w / img.width).toFixed(6)} ${(r.h / img.height).toFixed(6)}`;
    });
    return { filename: img.name.replace(/\.[^.]+$/, ".txt"), content: lines.join("\n") };
  });
  files.push({ filename: "classes.txt", content: allLabels.join("\n") });
  return files;
};

const exportToCOCO = (images, regions) => {
  const categories = [], categoryMap = {}, annotations = [];
  let annotationId = 1;
  const cocoImages = images.map((img, idx) => {
    (regions[img.id] || []).forEach((r) => {
      if (!categoryMap[r.label]) {
        const catId = categories.length + 1;
        categoryMap[r.label] = catId;
        categories.push({ id: catId, name: r.label, supercategory: "object" });
      }
      annotations.push({ id: annotationId++, image_id: idx + 1, category_id: categoryMap[r.label], bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h)], area: Math.round(r.w * r.h), iscrowd: 0, score: r.confidence });
    });
    return { id: idx + 1, file_name: img.name, width: img.width, height: img.height };
  });
  return { info: { description: "AutoLabel Pro Export", version: "1.0", year: new Date().getFullYear() }, images: cocoImages, annotations, categories };
};

const exportToVOC = (images, regions) => images.map((img) => {
  const objects = (regions[img.id] || []).map((r) => `\n    <object><n>${r.label}</n><bndbox><xmin>${Math.round(r.x)}</xmin><ymin>${Math.round(r.y)}</ymin><xmax>${Math.round(r.x + r.w)}</xmax><ymax>${Math.round(r.y + r.h)}</ymax></bndbox><confidence>${r.confidence.toFixed(3)}</confidence></object>`).join("");
  return { filename: img.name.replace(/\.[^.]+$/, ".xml"), content: `<?xml version="1.0"?>\n<annotation><filename>${img.name}</filename><size><width>${img.width}</width><height>${img.height}</height><depth>3</depth></size>${objects}\n</annotation>` };
});

// ===== COMPONENTS =====
const Toast = ({ message, type, onClose }) => {
  useEffect(() => { const t = setTimeout(onClose, 4000); return () => clearTimeout(t); }, [onClose]);
  const bg = { success: "#38A169", warning: "#D69E2E", error: "#E53E3E", info: PRIMARY_COLOR }[type];
  return (
    <div style={{ position: "fixed", bottom: 24, left: "50%", transform: "translateX(-50%)", padding: "12px 20px", background: bg, color: "#FFF", borderRadius: 8, fontSize: 13, boxShadow: "0 4px 20px rgba(0,0,0,0.25)", display: "flex", alignItems: "center", gap: 10, zIndex: 1000 }}>
      <span>{message}</span>
      <button onClick={onClose} style={{ background: "none", border: "none", color: "#FFF", cursor: "pointer", fontSize: 16 }}>×</button>
    </div>
  );
};

const ConfirmModal = ({ isOpen, onClose, onConfirm, title, message, confirmText, confirmColor }) => {
  if (!isOpen) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }} onClick={onClose}>
      <div style={{ background: "#FFF", borderRadius: 12, padding: 24, width: 400, boxShadow: "0 20px 60px rgba(0,0,0,0.3)" }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ margin: "0 0 12px", fontSize: 18 }}>{title}</h3>
        <p style={{ margin: "0 0 20px", color: "#4A5568", fontSize: 14, lineHeight: 1.5 }}>{message}</p>
        <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
          <button onClick={onClose} style={{ padding: "10px 20px", background: "#FFF", border: "1px solid #E2E8F0", borderRadius: 8, cursor: "pointer", fontSize: 14 }}>Cancel</button>
          <button onClick={() => { onConfirm(); onClose(); }} style={{ padding: "10px 24px", background: confirmColor || "#E53E3E", color: "#FFF", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600, fontSize: 14 }}>{confirmText || "Confirm"}</button>
        </div>
      </div>
    </div>
  );
};

const ExportModal = ({ isOpen, onClose, onExport, totalLabels }) => {
  const [format, setFormat] = useState("yolo");
  if (!isOpen) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }} onClick={onClose}>
      <div style={{ background: "#FFF", borderRadius: 12, padding: 24, width: 420, boxShadow: "0 20px 60px rgba(0,0,0,0.3)" }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ margin: "0 0 16px", fontSize: 18 }}>Export {totalLabels} Labels</h3>
        <div style={{ marginBottom: 20 }}>
          {Object.entries(EXPORT_FORMATS).map(([key, fmt]) => (
            <label key={key} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", marginBottom: 8, border: `2px solid ${format === key ? PRIMARY_COLOR : "#E2E8F0"}`, borderRadius: 8, cursor: "pointer", background: format === key ? PRIMARY_LIGHT : "#FFF" }}>
              <input type="radio" checked={format === key} onChange={() => setFormat(key)} />
              <div><div style={{ fontWeight: 600 }}>{fmt.name}</div><div style={{ fontSize: 11, color: "#718096" }}>{fmt.description}</div></div>
            </label>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
          <button onClick={onClose} style={{ padding: "10px 20px", background: "#FFF", border: "1px solid #E2E8F0", borderRadius: 8, cursor: "pointer" }}>Cancel</button>
          <button onClick={() => onExport(format)} style={{ padding: "10px 24px", background: PRIMARY_COLOR, color: "#FFF", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 600 }}>Export</button>
        </div>
      </div>
    </div>
  );
};

const LabelEditorModal = ({ isOpen, onClose, onSave, currentLabel, existingLabels }) => {
  const [label, setLabel] = useState(currentLabel || "");
  const inputRef = useRef(null);
  useEffect(() => { if (isOpen) { setLabel(currentLabel || ""); setTimeout(() => inputRef.current?.focus(), 50); } }, [isOpen, currentLabel]);
  if (!isOpen) return null;
  const handleSubmit = (e) => { e.preventDefault(); if (label.trim()) onSave(label.trim()); };
  const filteredSuggestions = existingLabels.filter((l) => l.toLowerCase().includes(label.toLowerCase()) && l !== label);
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }} onClick={onClose}>
      <div style={{ background: "#FFF", borderRadius: 12, padding: 24, width: 360, boxShadow: "0 20px 60px rgba(0,0,0,0.3)" }} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ margin: "0 0 16px", fontSize: 16 }}>Edit Label</h3>
        <form onSubmit={handleSubmit}>
          <input ref={inputRef} type="text" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Enter class name..." style={{ width: "100%", padding: "10px 12px", fontSize: 14, border: "1px solid #E2E8F0", borderRadius: 6, marginBottom: 8, boxSizing: "border-box" }} />
          {filteredSuggestions.length > 0 && (<div style={{ marginBottom: 12 }}><div style={{ fontSize: 11, color: "#718096", marginBottom: 6 }}>Existing classes:</div><div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>{filteredSuggestions.slice(0, 8).map((s) => (<button key={s} type="button" onClick={() => setLabel(s)} style={{ padding: "4px 10px", fontSize: 11, background: "#F7FAFC", border: "1px solid #E2E8F0", borderRadius: 4, cursor: "pointer" }}>{s}</button>))}</div></div>)}
          <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
            <button type="button" onClick={onClose} style={{ padding: "8px 16px", background: "#FFF", border: "1px solid #E2E8F0", borderRadius: 6, cursor: "pointer" }}>Cancel</button>
            <button type="submit" disabled={!label.trim()} style={{ padding: "8px 20px", background: label.trim() ? PRIMARY_COLOR : "#E2E8F0", color: label.trim() ? "#FFF" : "#A0AEC0", border: "none", borderRadius: 6, cursor: label.trim() ? "pointer" : "not-allowed", fontWeight: 600 }}>Save</button>
          </div>
        </form>
      </div>
    </div>
  );
};

// Context Menu Component (NEW)
const ContextMenu = ({ x, y, onClose, onChangeLabel, onDelete, selectedCount, existingLabels }) => {
  const [showLabelSubmenu, setShowLabelSubmenu] = useState(false);
  
  useEffect(() => {
    const handleClick = () => onClose();
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, [onClose]);

  return (
    <div
      style={{
        position: "fixed",
        left: x,
        top: y,
        background: "#FFF",
        borderRadius: 8,
        boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
        border: "1px solid #E2E8F0",
        minWidth: 180,
        zIndex: 1000,
        overflow: "hidden",
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div style={{ padding: "8px 12px", borderBottom: "1px solid #E2E8F0", fontSize: 11, color: "#718096" }}>
        {selectedCount} item{selectedCount > 1 ? "s" : ""} selected
      </div>
      <div
        style={{ position: "relative" }}
        onMouseEnter={() => setShowLabelSubmenu(true)}
        onMouseLeave={() => setShowLabelSubmenu(false)}
      >
        <div style={{ padding: "10px 12px", cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}
          onMouseEnter={(e) => e.target.style.background = "#F7FAFC"}
          onMouseLeave={(e) => e.target.style.background = "transparent"}
        >
          <span>Change Label to...</span>
          <span style={{ color: "#A0AEC0" }}>▶</span>
        </div>
        {showLabelSubmenu && (
          <div style={{ position: "absolute", left: "100%", top: 0, background: "#FFF", borderRadius: 8, boxShadow: "0 4px 20px rgba(0,0,0,0.15)", border: "1px solid #E2E8F0", minWidth: 140 }}>
            {existingLabels.slice(0, 10).map((label) => (
              <div
                key={label}
                onClick={() => { onChangeLabel(label); onClose(); }}
                style={{ padding: "8px 12px", cursor: "pointer", fontSize: 12, display: "flex", alignItems: "center", gap: 8 }}
                onMouseEnter={(e) => e.target.style.background = "#F7FAFC"}
                onMouseLeave={(e) => e.target.style.background = "transparent"}
              >
                <span style={{ width: 10, height: 10, borderRadius: 2, background: getClassColor(label) }} />
                {label}
              </div>
            ))}
          </div>
        )}
      </div>
      <div
        onClick={() => { onDelete(); onClose(); }}
        style={{ padding: "10px 12px", cursor: "pointer", color: "#E53E3E", fontSize: 13, display: "flex", alignItems: "center", gap: 8 }}
        onMouseEnter={(e) => e.target.style.background = "#FED7D7"}
        onMouseLeave={(e) => e.target.style.background = "transparent"}
      >
        <span>🗑</span> Delete Selected
      </div>
    </div>
  );
};

const ResizeHandles = ({ box, onResizeStart, color }) => {
  const handles = [
    { pos: "nw", cursor: "nwse-resize", x: -HANDLE_SIZE / 2, y: -HANDLE_SIZE / 2 },
    { pos: "n", cursor: "ns-resize", x: box.w / 2 - HANDLE_SIZE / 2, y: -HANDLE_SIZE / 2 },
    { pos: "ne", cursor: "nesw-resize", x: box.w - HANDLE_SIZE / 2, y: -HANDLE_SIZE / 2 },
    { pos: "e", cursor: "ew-resize", x: box.w - HANDLE_SIZE / 2, y: box.h / 2 - HANDLE_SIZE / 2 },
    { pos: "se", cursor: "nwse-resize", x: box.w - HANDLE_SIZE / 2, y: box.h - HANDLE_SIZE / 2 },
    { pos: "s", cursor: "ns-resize", x: box.w / 2 - HANDLE_SIZE / 2, y: box.h - HANDLE_SIZE / 2 },
    { pos: "sw", cursor: "nesw-resize", x: -HANDLE_SIZE / 2, y: box.h - HANDLE_SIZE / 2 },
    { pos: "w", cursor: "ew-resize", x: -HANDLE_SIZE / 2, y: box.h / 2 - HANDLE_SIZE / 2 },
  ];
  return (<>{handles.map((h) => (<div key={h.pos} onMouseDown={(e) => { e.stopPropagation(); onResizeStart(h.pos, e); }} style={{ position: "absolute", left: h.x, top: h.y, width: HANDLE_SIZE, height: HANDLE_SIZE, background: "#FFF", border: `2px solid ${color}`, borderRadius: 2, cursor: h.cursor, zIndex: 10 }} />))}</>);
};

// Button with Tooltip Component (NEW)
const TooltipButton = ({ onClick, disabled, style, children, shortcut, title }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={onClick}
        disabled={disabled}
        style={style}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {children}
        {shortcut && <span style={{ marginLeft: 4, opacity: 0.7, fontSize: "0.85em" }}>{shortcut}</span>}
      </button>
      {showTooltip && title && (
        <div style={{
          position: "absolute",
          bottom: "100%",
          left: "50%",
          transform: "translateX(-50%)",
          marginBottom: 6,
          padding: "4px 8px",
          background: "#1A202C",
          color: "#FFF",
          fontSize: 10,
          borderRadius: 4,
          whiteSpace: "nowrap",
          zIndex: 1000,
        }}>
          {title}
        </div>
      )}
    </div>
  );
};

// Filter Chip Component
const FilterChip = ({ label, count, color, isActive, onClick }) => (
  <button onClick={onClick} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", fontSize: 11, fontWeight: 500, background: isActive ? (color ? `${color}20` : PRIMARY_LIGHT) : "#F7FAFC", color: isActive ? (color || PRIMARY_COLOR) : "#718096", border: `1px solid ${isActive ? (color || PRIMARY_COLOR) : "#E2E8F0"}`, borderRadius: 16, cursor: "pointer", transition: "all 0.15s" }}>
    {color && <span style={{ width: 8, height: 8, borderRadius: 2, background: color }} />}
    <span>{label}</span>
    <span style={{ background: isActive ? (color || PRIMARY_COLOR) : "#E2E8F0", color: isActive ? "#FFF" : "#718096", padding: "1px 6px", borderRadius: 10, fontSize: 10 }}>{count}</span>
  </button>
);

// Category Accordion Component
const CategoryAccordion = ({ label, color, regions, isExpanded, onToggle, selectedRegions, onSelectRegion, onDoubleClick, hoveredRegion, setHoveredRegion, activeFilter, lastSelectedIndex, setLastSelectedIndex, allRegionIds }) => {
  const approvedCount = regions.filter(r => r.status === "approved").length;
  const flaggedCount = regions.filter(r => r.status === "flagged").length;
  
  return (
    <div style={{ marginBottom: 4 }}>
      <div onClick={onToggle} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", background: isExpanded ? "#F7FAFC" : "#FFF", border: "1px solid #E2E8F0", borderRadius: isExpanded ? "6px 6px 0 0" : 6, cursor: "pointer", userSelect: "none" }}>
        <span style={{ fontSize: 10, color: "#718096", transition: "transform 0.2s", transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)" }}>▶</span>
        <span style={{ width: 12, height: 12, borderRadius: 3, background: color, flexShrink: 0 }} />
        <span style={{ flex: 1, fontSize: 12, fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: 11, color: "#718096" }}>{regions.length}</span>
        {approvedCount > 0 && <span style={{ fontSize: 9, padding: "2px 5px", background: "#C6F6D520", color: "#38A169", borderRadius: 8 }}>✓{approvedCount}</span>}
        {flaggedCount > 0 && <span style={{ fontSize: 9, padding: "2px 5px", background: "#FEFCBF20", color: "#D69E2E", borderRadius: 8 }}>⚠{flaggedCount}</span>}
      </div>
      {isExpanded && (
        <div style={{ border: "1px solid #E2E8F0", borderTop: "none", borderRadius: "0 0 6px 6px", background: "#FFF" }}>
          {regions.map((region, idx) => {
            const isSelected = selectedRegions.includes(region.id);
            const isHovered = hoveredRegion === region.id;
            const status = REVIEW_STATUS[region.status] || REVIEW_STATUS.auto;
            const isDimmed = activeFilter && activeFilter !== label;
            const globalIdx = allRegionIds.indexOf(region.id);
            
            return (
              <div
                key={region.id}
                onClick={(e) => onSelectRegion(region.id, e, globalIdx)}
                onDoubleClick={(e) => onDoubleClick(region.id, e)}
                onMouseEnter={() => setHoveredRegion(region.id)}
                onMouseLeave={() => setHoveredRegion(null)}
                style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px 6px 28px", cursor: "pointer", background: isSelected ? PRIMARY_LIGHT : isHovered ? "#F7FAFC" : "#FFF", borderBottom: "1px solid #F0F0F0", opacity: isDimmed ? 0.4 : 1 }}
              >
                <span style={{ flex: 1, fontSize: 11, color: "#4A5568" }}>{region.label}</span>
                <span style={{ fontSize: 10, color: region.confidence < LOW_CONFIDENCE_THRESHOLD ? "#E53E3E" : "#A0AEC0" }}>
                  {(region.confidence * 100).toFixed(0)}%
                </span>
                <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 8, background: `${status.color}15`, color: status.color }}>{status.icon || status.label}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ===== MAIN COMPONENT =====
export default function AutoLabelProV7() {
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  // Server config
  const [serverUrls, setServerUrls] = useState({ "grounding-dino": "http://localhost:8000", "yolo-v8": "http://localhost:8001", "yolo-v11": "http://localhost:8001" });
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
  const [labelDisplayMode, setLabelDisplayMode] = useState("always");
  const [toast, setToast] = useState(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Collapsible controls (NEW)
  const [controlsCollapsed, setControlsCollapsed] = useState(false);

  // Context menu state (NEW)
  const [contextMenu, setContextMenu] = useState(null);
  const [lastSelectedIndex, setLastSelectedIndex] = useState(null);

  // Filter state
  const [activeFilter, setActiveFilter] = useState(null);
  const [expandedCategories, setExpandedCategories] = useState(new Set());

  // Confirmation modals
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showDetectOverwriteConfirm, setShowDetectOverwriteConfirm] = useState(false);
  const [pendingDetectAction, setPendingDetectAction] = useState(null);

  // Annotation mode state
  const [annotationMode, setAnnotationMode] = useState("select");
  const [defaultLabel, setDefaultLabel] = useState("object");

  // Drawing state
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [drawCurrent, setDrawCurrent] = useState(null);

  // Dragging state
  const [isDragging, setIsDragging] = useState(false);
  const [dragRegionId, setDragRegionId] = useState(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Resizing state
  const [isResizing, setIsResizing] = useState(false);
  const [resizeRegionId, setResizeRegionId] = useState(null);
  const [resizeHandle, setResizeHandle] = useState(null);
  const [resizeStart, setResizeStart] = useState(null);

  // Label editor state
  const [showLabelEditor, setShowLabelEditor] = useState(false);
  const [editingRegionId, setEditingRegionId] = useState(null);

  // Undo/Redo state
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

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
  const existingLabels = [...new Set(Object.values(regions).flat().map((r) => r.label))];
  const allRegionIds = currentRegions.map(r => r.id);

  // Progress calculation (NEW)
  const progressStats = useMemo(() => {
    let reviewed = 0, detected = 0, todo = 0;
    images.forEach((img) => {
      const status = getImageStatus(regions[img.id]);
      if (status === "reviewed") reviewed++;
      else if (status === "detected") detected++;
      else todo++;
    });
    return { reviewed, detected, todo, total: images.length };
  }, [images, regions]);

  // Group regions by label
  const groupedRegions = useMemo(() => {
    const groups = {};
    currentRegions.forEach((region) => {
      if (!groups[region.label]) groups[region.label] = [];
      groups[region.label].push(region);
    });
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [currentRegions]);

  // Filter chips data
  const filterChips = useMemo(() => [
    { label: "All", count: currentRegions.length, color: null },
    ...groupedRegions.map(([label, items]) => ({ label, count: items.length, color: getClassColor(label) })),
  ], [currentRegions, groupedRegions]);

  const showToast = useCallback((msg, type = "success") => setToast({ message: msg, type }), []);

  const saveToHistory = useCallback((newRegions) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(JSON.stringify(newRegions));
    if (newHistory.length > 50) newHistory.shift();
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  }, [history, historyIndex]);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) { setHistoryIndex(historyIndex - 1); setRegions(JSON.parse(history[historyIndex - 1])); showToast("Undo", "info"); }
  }, [historyIndex, history, showToast]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) { setHistoryIndex(historyIndex + 1); setRegions(JSON.parse(history[historyIndex + 1])); showToast("Redo", "info"); }
  }, [historyIndex, history, showToast]);

  useEffect(() => {
    if (currentModelConfig?.thresholds) {
      const defaults = {};
      Object.entries(currentModelConfig.thresholds).forEach(([k, v]) => { defaults[k] = v.default; });
      setThresholds(defaults);
    }
  }, [selectedModel]);

  useEffect(() => {
    const check = async () => {
      setModelStatus("checking");
      const result = await checkModelHealth(currentServerUrl, currentModelConfig?.healthEndpoint || "/health");
      setModelStatus(result.status === "connected" ? "connected" : "disconnected");
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, [selectedModel, currentServerUrl]);

  useEffect(() => {
    if (groupedRegions.length > 0 && expandedCategories.size === 0) {
      setExpandedCategories(new Set([groupedRegions[0][0]]));
    }
  }, [groupedRegions]);

  const applySensitivityPreset = (key) => {
    setSensitivityPreset(key);
    const preset = SENSITIVITY_PRESETS[key];
    if (currentModelConfig?.thresholds) {
      const newT = {};
      Object.entries(currentModelConfig.thresholds).forEach(([k, v]) => { newT[k] = Math.max(v.min, Math.min(v.max, v.default * preset.multiplier)); });
      setThresholds(newT);
    }
  };

  const handleImportClick = () => fileInputRef.current?.click();

  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    const newImages = await Promise.all(files.map(async (file, idx) => {
      const url = URL.createObjectURL(file);
      const base64 = await imageToBase64(file);
      const img = new Image();
      await new Promise((r) => { img.onload = r; img.src = url; });
      return { id: `img_${Date.now()}_${idx}`, name: file.name, url, base64, width: img.width, height: img.height };
    }));
    setImages((prev) => [...prev, ...newImages]);
    if (currentStep === 1) setCurrentStep(2);
    showToast(`${files.length} image(s) imported`, "success");
    e.target.value = "";
  };

  const initiateDetection = (type) => {
    if (!currentImage && type === 'current') { showToast("Import images first", "warning"); return; }
    if (images.length === 0 && type === 'all') { showToast("Import images first", "warning"); return; }
    if (modelStatus !== "connected") { showToast(`${currentModelConfig.name} not connected. Check server at ${currentServerUrl}`, "error"); return; }
    if (currentModelConfig.requiresPrompt && !prompt.trim()) { showToast("Enter detection prompt", "warning"); return; }
    const hasExisting = type === 'current' ? currentRegions.length > 0 : Object.values(regions).flat().length > 0;
    if (hasExisting) { setPendingDetectAction(type); setShowDetectOverwriteConfirm(true); }
    else { type === 'current' ? executeDetect() : executeDetectAll(); }
  };

  const executeDetect = async () => {
    setIsDetecting(true);
    try {
      const requestBody = selectedModel === "grounding-dino"
        ? { image: currentImage.base64, prompt, box_threshold: thresholds.box || 0.35, text_threshold: thresholds.text || 0.25, nms_threshold: thresholds.nms || 0.5 }
        : { image: currentImage.base64, confidence_threshold: thresholds.confidence || 0.25, iou_threshold: thresholds.iou || 0.45 };
      const result = await runDetection(currentServerUrl, currentModelConfig.endpoint, requestBody);
      const detections = (result.detections || []).map((det, idx) => ({
        id: Date.now() + idx, label: det.label || det.class || "object", confidence: det.confidence || det.score || 0,
        status: "auto", x: det.bbox?.[0] || 0, y: det.bbox?.[1] || 0, w: det.bbox?.[2] || 100, h: det.bbox?.[3] || 100,
      }));
      const newRegions = { ...regions, [currentImage.id]: detections };
      setRegions(newRegions);
      saveToHistory(newRegions);
      setCurrentStep(3);
      setActiveFilter(null);
      setExpandedCategories(new Set());
      setControlsCollapsed(true); // Auto-collapse after detection
      showToast(`${detections.length} objects detected`, "success");
    } catch (err) { showToast(`Detection failed: ${err.message}`, "error"); }
    finally { setIsDetecting(false); }
  };

  const executeDetectAll = async () => {
    setIsDetecting(true);
    let totalDetected = 0, processedCount = 0;
    const allNewRegions = {};
    try {
      for (const img of images) {
        processedCount++;
        showToast(`Processing ${processedCount}/${images.length}: ${img.name}`, "info");
        const requestBody = selectedModel === "grounding-dino"
          ? { image: img.base64, prompt, box_threshold: thresholds.box || 0.35, text_threshold: thresholds.text || 0.25, nms_threshold: thresholds.nms || 0.5 }
          : { image: img.base64, confidence_threshold: thresholds.confidence || 0.25, iou_threshold: thresholds.iou || 0.45 };
        try {
          const result = await runDetection(currentServerUrl, currentModelConfig.endpoint, requestBody);
          const detections = (result.detections || []).map((det, idx) => ({
            id: Date.now() + idx + processedCount * 10000, label: det.label || det.class || "object",
            confidence: det.confidence || det.score || 0, status: "auto",
            x: det.bbox?.[0] || 0, y: det.bbox?.[1] || 0, w: det.bbox?.[2] || 100, h: det.bbox?.[3] || 100,
          }));
          allNewRegions[img.id] = detections;
          totalDetected += detections.length;
        } catch (err) { console.error(`Error detecting ${img.name}:`, err); }
      }
      const newRegions = { ...regions, ...allNewRegions };
      setRegions(newRegions);
      saveToHistory(newRegions);
      setCurrentStep(3);
      setControlsCollapsed(true);
      showToast(`Completed! Detected ${totalDetected} objects in ${images.length} images`, "success");
    } catch (err) { showToast(`Detection failed: ${err.message}`, "error"); }
    finally { setIsDetecting(false); }
  };

  const handleConfirmDetect = () => {
    if (pendingDetectAction === 'current') executeDetect();
    else if (pendingDetectAction === 'all') executeDetectAll();
    setPendingDetectAction(null);
  };

  const handleExport = (format) => {
    setShowExportModal(false);
    if (totalRegionsAllImages === 0) { showToast("No labels to export", "warning"); return; }
    const ts = Date.now();
    try {
      if (format === "yolo") {
        const files = exportToYOLO(images, regions);
        const content = files.map((f) => `=== ${f.filename} ===\n${f.content}`).join("\n\n");
        downloadFile(content, `yolo_labels_${ts}.txt`, "text/plain");
      } else if (format === "coco") {
        downloadFile(JSON.stringify(exportToCOCO(images, regions), null, 2), `coco_${ts}.json`, "application/json");
      } else if (format === "voc") {
        const files = exportToVOC(images, regions);
        const content = files.map((f) => `=== ${f.filename} ===\n${f.content}`).join("\n\n");
        downloadFile(content, `voc_${ts}.xml`, "text/plain");
      } else {
        const data = { images: images.map((img) => ({ filename: img.name, width: img.width, height: img.height, annotations: (regions[img.id] || []).map((r) => ({ label: r.label, confidence: r.confidence, status: r.status, bbox: { x: r.x, y: r.y, w: r.w, h: r.h } })) })) };
        downloadFile(JSON.stringify(data, null, 2), `annotations_${ts}.json`, "application/json");
      }
      setCurrentStep(4);
      showToast(`Exported ${totalRegionsAllImages} labels in ${format.toUpperCase()} format`, "success");
    } catch (err) { showToast(`Export failed: ${err.message}`, "error"); }
  };

  const downloadFile = (content, filename, type) => {
    const blob = new Blob([content], { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const updateRegions = (fn) => {
    if (!currentImage) return;
    const newRegions = { ...regions, [currentImage.id]: fn(regions[currentImage.id] || []) };
    setRegions(newRegions);
    saveToHistory(newRegions);
  };

  const handleApprove = () => {
    if (selectedRegions.length) {
      updateRegions((rs) => rs.map((r) => selectedRegions.includes(r.id) ? { ...r, status: "approved" } : r));
      showToast(`${selectedRegions.length} approved`, "success");
      setSelectedRegions([]);
    }
  };

  const handleFlag = () => {
    if (selectedRegions.length) {
      updateRegions((rs) => rs.map((r) => selectedRegions.includes(r.id) ? { ...r, status: "flagged" } : r));
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

  // Bulk change label (NEW)
  const handleBulkChangeLabel = (newLabel) => {
    if (selectedRegions.length) {
      updateRegions((rs) => rs.map((r) => selectedRegions.includes(r.id) ? { ...r, label: newLabel } : r));
      showToast(`Changed ${selectedRegions.length} items to "${newLabel}"`, "success");
      setSelectedRegions([]);
    }
  };

  const handleDeleteImage = (imgId, e) => {
    e?.stopPropagation();
    const imgIndex = images.findIndex((img) => img.id === imgId);
    const imgName = images[imgIndex]?.name;
    setImages((prev) => prev.filter((img) => img.id !== imgId));
    const newRegions = { ...regions };
    delete newRegions[imgId];
    setRegions(newRegions);
    if (imgIndex <= currentImageIndex && currentImageIndex > 0) setCurrentImageIndex((prev) => prev - 1);
    setSelectedRegions([]);
    showToast(`Removed "${imgName}"`, "info");
  };

  const handleClearCurrentDetections = () => {
    if (!currentImage) return;
    if (currentRegions.length === 0) { showToast("No detections to clear", "warning"); return; }
    setShowClearConfirm(true);
  };

  const executeClearCurrentDetections = () => {
    const count = currentRegions.length;
    const newRegions = { ...regions, [currentImage.id]: [] };
    setRegions(newRegions);
    saveToHistory(newRegions);
    setSelectedRegions([]);
    setActiveFilter(null);
    showToast(`Cleared ${count} detections`, "info");
  };

  // Enhanced select with Shift+Click support (NEW)
  const handleSelectRegion = (id, e, globalIdx) => {
    if (annotationMode === "draw") return;
    
    if (e?.shiftKey && lastSelectedIndex !== null && globalIdx !== undefined) {
      // Shift+Click: range selection
      const start = Math.min(lastSelectedIndex, globalIdx);
      const end = Math.max(lastSelectedIndex, globalIdx);
      const rangeIds = allRegionIds.slice(start, end + 1);
      setSelectedRegions((prev) => [...new Set([...prev, ...rangeIds])]);
    } else if (e?.metaKey || e?.ctrlKey) {
      // Ctrl/Cmd+Click: toggle selection
      setSelectedRegions((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
      setLastSelectedIndex(globalIdx);
    } else {
      // Normal click: single selection
      setSelectedRegions([id]);
      setLastSelectedIndex(globalIdx);
    }
  };

  // Right-click context menu (NEW)
  const handleContextMenu = (e) => {
    if (selectedRegions.length > 0) {
      e.preventDefault();
      setContextMenu({ x: e.clientX, y: e.clientY });
    }
  };

  const toggleCategory = (label) => {
    setExpandedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(label)) newSet.delete(label);
      else newSet.add(label);
      return newSet;
    });
  };

  const handleFilterClick = (label) => {
    if (label === "All") setActiveFilter(null);
    else setActiveFilter(activeFilter === label ? null : label);
  };

  const getCanvasCoords = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    return { x: (e.clientX - centerX) / zoom + CANVAS_WIDTH / 2, y: (e.clientY - centerY) / zoom + CANVAS_HEIGHT / 2 };
  };

  const handleCanvasMouseDown = (e) => {
    if (!currentImage) return;
    const canvasCoords = getCanvasCoords(e);
    if (annotationMode === "draw") {
      setIsDrawing(true); setDrawStart(canvasCoords); setDrawCurrent(canvasCoords); setSelectedRegions([]);
    } else if (annotationMode === "select" || annotationMode === "edit") {
      const clickedRegion = currentRegions.find((r) => {
        const scaled = getScaledBoundingBox(r, currentImage.width, currentImage.height);
        return canvasCoords.x >= scaled.x && canvasCoords.x <= scaled.x + scaled.w && canvasCoords.y >= scaled.y && canvasCoords.y <= scaled.y + scaled.h;
      });
      if (!clickedRegion) setSelectedRegions([]);
    }
  };

  const handleCanvasMouseMove = (e) => {
    if (!currentImage) return;
    const canvasCoords = getCanvasCoords(e);
    if (isDrawing) { setDrawCurrent(canvasCoords); }
    else if (isDragging && dragRegionId) {
      const imageCoords = canvasToImageCoords(canvasCoords.x - dragOffset.x, canvasCoords.y - dragOffset.y, currentImage.width, currentImage.height);
      setRegions((prev) => ({ ...prev, [currentImage.id]: (prev[currentImage.id] || []).map((r) => r.id === dragRegionId ? { ...r, x: clamp(imageCoords.x, 0, currentImage.width - r.w), y: clamp(imageCoords.y, 0, currentImage.height - r.h) } : r) }));
    } else if (isResizing && resizeRegionId && resizeHandle && resizeStart) {
      const region = currentRegions.find((r) => r.id === resizeRegionId);
      if (!region) return;
      const imageCoords = canvasToImageCoords(canvasCoords.x, canvasCoords.y, currentImage.width, currentImage.height);
      let newX = region.x, newY = region.y, newW = region.w, newH = region.h;
      if (resizeHandle.includes("w")) { const deltaX = imageCoords.x - resizeStart.x; newX = clamp(region.x + deltaX, 0, region.x + region.w - 20); newW = region.w - (newX - region.x); }
      if (resizeHandle.includes("e")) { newW = clamp(imageCoords.x - region.x, 20, currentImage.width - region.x); }
      if (resizeHandle.includes("n")) { const deltaY = imageCoords.y - resizeStart.y; newY = clamp(region.y + deltaY, 0, region.y + region.h - 20); newH = region.h - (newY - region.y); }
      if (resizeHandle.includes("s")) { newH = clamp(imageCoords.y - region.y, 20, currentImage.height - region.y); }
      setRegions((prev) => ({ ...prev, [currentImage.id]: (prev[currentImage.id] || []).map((r) => r.id === resizeRegionId ? { ...r, x: newX, y: newY, w: newW, h: newH } : r) }));
      setResizeStart(imageCoords);
    }
  };

  const handleCanvasMouseUp = () => {
    if (isDrawing && drawStart && drawCurrent && currentImage) {
      const startImg = canvasToImageCoords(drawStart.x, drawStart.y, currentImage.width, currentImage.height);
      const endImg = canvasToImageCoords(drawCurrent.x, drawCurrent.y, currentImage.width, currentImage.height);
      const x = Math.min(startImg.x, endImg.x), y = Math.min(startImg.y, endImg.y);
      const w = Math.abs(endImg.x - startImg.x), h = Math.abs(endImg.y - startImg.y);
      if (w > 10 && h > 10) {
        const newRegion = { id: Date.now(), label: defaultLabel, confidence: 1.0, status: "manual", x: clamp(x, 0, currentImage.width - w), y: clamp(y, 0, currentImage.height - h), w: Math.min(w, currentImage.width), h: Math.min(h, currentImage.height) };
        const newRegions = { ...regions, [currentImage.id]: [...(regions[currentImage.id] || []), newRegion] };
        setRegions(newRegions); saveToHistory(newRegions); setSelectedRegions([newRegion.id]);
        showToast(`Created "${defaultLabel}" annotation`, "success");
        setEditingRegionId(newRegion.id); setShowLabelEditor(true);
      }
      setIsDrawing(false); setDrawStart(null); setDrawCurrent(null);
    }
    if (isDragging) { saveToHistory(regions); setIsDragging(false); setDragRegionId(null); setDragOffset({ x: 0, y: 0 }); }
    if (isResizing) { saveToHistory(regions); setIsResizing(false); setResizeRegionId(null); setResizeHandle(null); setResizeStart(null); }
  };

  const handleDragStart = (regionId, e) => {
    if (annotationMode !== "edit") return;
    e.stopPropagation();
    const canvasCoords = getCanvasCoords(e);
    const region = currentRegions.find((r) => r.id === regionId);
    if (!region) return;
    const scaled = getScaledBoundingBox(region, currentImage.width, currentImage.height);
    setIsDragging(true); setDragRegionId(regionId);
    setDragOffset({ x: canvasCoords.x - scaled.x, y: canvasCoords.y - scaled.y });
    setSelectedRegions([regionId]);
  };

  const handleResizeStart = (regionId, handle, e) => {
    e.stopPropagation();
    const canvasCoords = getCanvasCoords(e);
    const imageCoords = canvasToImageCoords(canvasCoords.x, canvasCoords.y, currentImage.width, currentImage.height);
    setIsResizing(true); setResizeRegionId(regionId); setResizeHandle(handle); setResizeStart(imageCoords);
    setSelectedRegions([regionId]);
  };

  const handleDoubleClick = (regionId, e) => { e.stopPropagation(); setEditingRegionId(regionId); setShowLabelEditor(true); };

  const handleSaveLabel = (newLabel) => {
    if (editingRegionId) {
      updateRegions((rs) => rs.map((r) => r.id === editingRegionId ? { ...r, label: newLabel } : r));
      showToast(`Label updated to "${newLabel}"`, "success");
    }
    setShowLabelEditor(false); setEditingRegionId(null);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.25, 4));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.25, 0.5));
  const handleZoomReset = () => setZoom(1);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "v" || e.key === "V") { setAnnotationMode("select"); return; }
      if (e.key === "d" || e.key === "D") { setAnnotationMode("draw"); return; }
      if (e.key === "e" || e.key === "E") { setAnnotationMode("edit"); return; }
      if ((e.metaKey || e.ctrlKey) && e.key === "z") { e.preventDefault(); e.shiftKey ? handleRedo() : handleUndo(); return; }
      if (e.key === "ArrowLeft" && currentImageIndex > 0) { setCurrentImageIndex((i) => i - 1); setSelectedRegions([]); setActiveFilter(null); }
      if (e.key === "ArrowRight" && currentImageIndex < images.length - 1) { setCurrentImageIndex((i) => i + 1); setSelectedRegions([]); setActiveFilter(null); }
      if ((e.key === "Delete" || e.key === "Backspace") && selectedRegions.length) { e.preventDefault(); handleDelete(); }
      if (e.key === "Escape") { setSelectedRegions([]); setIsDrawing(false); setIsDragging(false); setIsResizing(false); setActiveFilter(null); setContextMenu(null); }
      if (e.key === "=" || e.key === "+") handleZoomIn();
      if (e.key === "-") handleZoomOut();
      if (e.key === "0") handleZoomReset();
      if (e.key === "Enter" && selectedRegions.length) { e.preventDefault(); handleApprove(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentImageIndex, images.length, selectedRegions, handleUndo, handleRedo]);

  const modelsByCategory = Object.values(MODEL_CONFIGS).reduce((acc, m) => { (acc[m.category] = acc[m.category] || []).push(m); return acc; }, {});

  const getCursor = () => {
    if (isDrawing) return "crosshair";
    if (isDragging) return "grabbing";
    if (isResizing) return "nwse-resize";
    return ANNOTATION_MODES[annotationMode]?.cursor || "default";
  };

  const shouldShowLabel = (regionId) => {
    if (labelDisplayMode === "always") return true;
    if (labelDisplayMode === "hidden") return false;
    if (labelDisplayMode === "hover") return hoveredRegion === regionId || selectedRegions.includes(regionId);
    return true;
  };

  const isRegionDimmed = (region) => activeFilter && region.label !== activeFilter;

  const editingRegion = currentRegions.find((r) => r.id === editingRegionId);

  const s = {
    container: { fontFamily: "'Inter', sans-serif", height: "97vh", display: "flex", flexDirection: "column", background: "#F7FAFC", overflow: "hidden" },
    header: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "1px 20px", background: "#FFF", borderBottom: "1px solid #E2E8F0" },
    stepper: { display: "flex", alignItems: "center", gap: 6 },
    step: (active, done) => ({ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 500, background: active ? PRIMARY_LIGHT : done ? "#F0FFF4" : "#F7FAFC", color: active ? "#2B6CB0" : done ? "#276749" : "#A0AEC0", border: active ? `1px solid ${PRIMARY_BORDER}` : "1px solid transparent", cursor: "pointer" }),
    stepNum: (active, done) => ({ width: 20, height: 20, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 600, background: done ? "#38A169" : active ? PRIMARY_COLOR : "#E2E8F0", color: done || active ? "#FFF" : "#A0AEC0" }),
    controls: { padding: "10px 20px", background: "#FFF", borderBottom: "1px solid #E2E8F0", display: "flex", gap: 16, alignItems: "flex-start" },
    main: { flex: 1, display: "flex", overflow: "hidden" },
    leftPanel: { width: 200, background: "#FFF", borderRight: "1px solid #E2E8F0", display: "flex", flexDirection: "column" },
    rightPanel: { width: 300, background: "#FFF", borderLeft: "1px solid #E2E8F0", display: "flex", flexDirection: "column" },
    canvas: { flex: 1, background: "#1A202C", display: "flex", flexDirection: "column" },
    canvasToolbar: { padding: "6px 12px", background: "#2D3748", display: "flex", alignItems: "center", justifyContent: "space-between" },
    canvasArea: { flex: 1, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", padding: 12 },
    btn: (primary, disabled) => ({ padding: "8px 16px", fontSize: 13, fontWeight: 600, borderRadius: 6, border: "none", cursor: disabled ? "not-allowed" : "pointer", background: disabled ? "#A0AEC0" : primary ? PRIMARY_COLOR : "#FFF", color: disabled ? "#FFF" : primary ? "#FFF" : "#4A5568" }),
    sectionTitle: { fontSize: 11, fontWeight: 600, color: "#718096", textTransform: "uppercase", marginBottom: 8 },
    modeButton: (active) => ({ display: "flex", alignItems: "center", justifyContent: "center", gap: 4, padding: "6px 10px", fontSize: 11, fontWeight: 500, background: active ? PRIMARY_COLOR : "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer", minWidth: 60 }),
  };

  return (
    <div style={s.container} onContextMenu={handleContextMenu}>
      <input ref={fileInputRef} type="file" accept="image/*" multiple style={{ display: "none" }} onChange={handleFileSelect} />
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
      <ExportModal isOpen={showExportModal} onClose={() => setShowExportModal(false)} onExport={handleExport} totalLabels={totalRegionsAllImages} />
      <LabelEditorModal isOpen={showLabelEditor} onClose={() => { setShowLabelEditor(false); setEditingRegionId(null); }} onSave={handleSaveLabel} currentLabel={editingRegion?.label} existingLabels={existingLabels} />
      <ConfirmModal isOpen={showClearConfirm} onClose={() => setShowClearConfirm(false)} onConfirm={executeClearCurrentDetections} title="Clear All Annotations?" message={`This will remove all ${currentRegions.length} annotations. Undo with Ctrl+Z.`} confirmText="Clear All" confirmColor="#E53E3E" />
      <ConfirmModal isOpen={showDetectOverwriteConfirm} onClose={() => { setShowDetectOverwriteConfirm(false); setPendingDetectAction(null); }} onConfirm={handleConfirmDetect} title="Overwrite Existing Annotations?" message={pendingDetectAction === 'current' ? `Replace ${currentRegions.length} annotation(s)?` : `Replace ${totalRegionsAllImages} annotations?`} confirmText="Replace & Detect" confirmColor={PRIMARY_COLOR} />
      {contextMenu && <ContextMenu x={contextMenu.x} y={contextMenu.y} onClose={() => setContextMenu(null)} onChangeLabel={handleBulkChangeLabel} onDelete={handleDelete} selectedCount={selectedRegions.length} existingLabels={existingLabels} />}

      {/* HEADER */}
      <header style={s.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 42, height: 42, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", color: "#0c0c0c", fontWeight: 700, fontSize: 20 }}>逢甲</div>
          <span style={{ fontSize: 16, fontWeight: 600 }}><p>自動駕駛研究團隊</p></span>
        </div>
        <div style={s.stepper}>
          {["Import", "Detect", "Review", "Export"].map((label, i) => (
            <React.Fragment key={label}>
              <div style={s.step(currentStep === i + 1, currentStep > i + 1)} onClick={() => { if (i + 1 <= currentStep || i === 0) setCurrentStep(i + 1); }}>
                <span style={s.stepNum(currentStep === i + 1, currentStep > i + 1)}>{currentStep > i + 1 ? "✓" : i + 1}</span>
                {label}
              </div>
              {i < 3 && <span style={{ color: "#CBD5E0" }}>›</span>}
            </React.Fragment>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <TooltipButton onClick={() => initiateDetection('current')} disabled={!currentImage || modelStatus !== "connected" || isDetecting} style={s.btn(true, !currentImage || modelStatus !== "connected" || isDetecting)} title="Run detection on current image">{isDetecting ? "Detecting..." : "▶ Detect"}</TooltipButton>
          <TooltipButton onClick={() => initiateDetection('all')} disabled={images.length === 0 || modelStatus !== "connected" || isDetecting} style={{ ...s.btn(false, images.length === 0 || modelStatus !== "connected" || isDetecting), background: images.length === 0 || modelStatus !== "connected" || isDetecting ? "#A0AEC0" : PRIMARY_LIGHT, color: images.length === 0 || modelStatus !== "connected" || isDetecting ? "#FFF" : "#2B6CB0" }} title="Run detection on all images">{`▶▶ All (${images.length})`}</TooltipButton>
        </div>
      </header>

      {/* COLLAPSIBLE CONTROLS (NEW) */}
      {!controlsCollapsed ? (
        <div style={s.controls}>
          <div style={{ minWidth: 180 }}>
            <div style={s.sectionTitle}>Detection Model</div>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} style={{ width: "100%", padding: "8px 10px", fontSize: 13, border: "1px solid #E2E8F0", borderRadius: 6 }}>
              {Object.entries(modelsByCategory).map(([cat, models]) => (<optgroup key={cat} label={cat}>{models.map((m) => (<option key={m.id} value={m.id} disabled={m.disabled}>{m.name}</option>))}</optgroup>))}
            </select>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, fontSize: 11 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: modelStatus === "connected" ? "#38A169" : modelStatus === "checking" ? "#D69E2E" : "#E53E3E" }} />
              <span style={{ color: modelStatus === "connected" ? "#38A169" : "#718096" }}>{modelStatus === "connected" ? "Connected" : modelStatus === "checking" ? "Checking..." : "Disconnected"}</span>
            </div>
            <button onClick={() => setShowServerConfig(!showServerConfig)} style={{ marginTop: 4, fontSize: 10, color: "#718096", background: "none", border: "none", cursor: "pointer" }}>⚙️ {showServerConfig ? "Hide" : "Server"}</button>
            {showServerConfig && <input type="text" value={currentServerUrl} onChange={(e) => setServerUrls((prev) => ({ ...prev, [selectedModel]: e.target.value }))} style={{ width: "100%", padding: "6px 8px", fontSize: 11, border: "1px solid #E2E8F0", borderRadius: 4, marginTop: 4 }} />}
          </div>
          {currentModelConfig?.requiresPrompt && (
            <div style={{ flex: 1, maxWidth: 500 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <div style={s.sectionTitle}>Detection Prompt</div>
                <div style={{ display: "flex", gap: 4 }}>{PROMPT_PRESETS.map((preset) => (<button key={preset.key} onClick={() => { setPrompt(preset.prompt); setSelectedPreset(preset.key); }} style={{ padding: "4px 10px", fontSize: 10, background: selectedPreset === preset.key ? PRIMARY_LIGHT : "#F7FAFC", border: `1px solid ${selectedPreset === preset.key ? PRIMARY_BORDER : "#E2E8F0"}`, borderRadius: 4, cursor: "pointer" }}>{preset.label}</button>))}</div>
              </div>
              <textarea value={prompt} onChange={(e) => { setPrompt(e.target.value); setSelectedPreset(null); }} placeholder="car . person . dog" style={{ width: "100%", height: 50, padding: "8px 10px", fontSize: 13, border: "1px solid #E2E8F0", borderRadius: 6, resize: "none", fontFamily: "inherit", boxSizing: "border-box" }} />
            </div>
          )}
          <div style={{ minWidth: 120 }}>
            <div style={s.sectionTitle}>Default Label</div>
            <input type="text" value={defaultLabel} onChange={(e) => setDefaultLabel(e.target.value)} style={{ width: "100%", padding: "8px 10px", fontSize: 13, border: "1px solid #E2E8F0", borderRadius: 6, boxSizing: "border-box" }} />
          </div>
          <button onClick={() => setControlsCollapsed(true)} style={{ padding: "8px", background: "none", border: "1px solid #E2E8F0", borderRadius: 6, cursor: "pointer", color: "#718096", fontSize: 12 }} title="Collapse detection controls">▲ Collapse</button>
        </div>
      ) : (
        <div style={{ padding: "6px 20px", background: "#FFF", borderBottom: "1px solid #E2E8F0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 12, color: "#718096" }}>
            <span><strong>Model:</strong> {currentModelConfig?.name}</span>
            {currentModelConfig?.requiresPrompt && <span><strong>Prompt:</strong> {prompt.substring(0, 30)}{prompt.length > 30 ? "..." : ""}</span>}
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: modelStatus === "connected" ? "#38A169" : "#E53E3E" }} />
              {modelStatus === "connected" ? "Connected" : "Disconnected"}
            </span>
          </div>
          <button onClick={() => setControlsCollapsed(false)} style={{ padding: "4px 12px", background: PRIMARY_LIGHT, border: `1px solid ${PRIMARY_BORDER}`, borderRadius: 4, cursor: "pointer", color: "#2B6CB0", fontSize: 11 }}>▼ Expand Detection Settings</button>
        </div>
      )}

      {/* MAIN */}
      <div style={s.main}>
        {/* LEFT PANEL */}
        <div style={s.leftPanel}>
          <div style={{ padding: "12px", borderBottom: "1px solid #E2E8F0" }}>
            <button onClick={handleImportClick} style={{ width: "100%", padding: "10px 16px", fontSize: 13, fontWeight: 600, background: PRIMARY_COLOR, color: "#FFF", border: "none", borderRadius: 6, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>📁 Import Images</button>
          </div>
          
          {/* Progress Bar (NEW) */}
          {images.length > 0 && (
            <div style={{ padding: "8px 12px", borderBottom: "1px solid #E2E8F0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 10, color: "#718096", marginBottom: 4 }}>
                <span>Progress</span>
                <span>{progressStats.reviewed}/{progressStats.total} done</span>
              </div>
              <div style={{ height: 6, background: "#E2E8F0", borderRadius: 3, overflow: "hidden", display: "flex" }}>
                <div style={{ width: `${(progressStats.reviewed / Math.max(1, progressStats.total)) * 100}%`, background: "#38A169", transition: "width 0.3s" }} />
                <div style={{ width: `${(progressStats.detected / Math.max(1, progressStats.total)) * 100}%`, background: "#D69E2E", transition: "width 0.3s" }} />
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 4, fontSize: 9 }}>
                <span style={{ color: "#38A169" }}>● {progressStats.reviewed} Done</span>
                <span style={{ color: "#D69E2E" }}>● {progressStats.detected} Detected</span>
                <span style={{ color: "#A0AEC0" }}>○ {progressStats.todo} Todo</span>
              </div>
            </div>
          )}

          <div style={{ padding: "10px 12px", borderBottom: "1px solid #E2E8F0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 600 }}>Images</span>
            <span style={{ padding: "2px 8px", background: "#EDF2F7", borderRadius: 10, fontSize: 11 }}>{images.length}</span>
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 8 }}>
            {images.length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "#A0AEC0" }}><div style={{ fontSize: 28, marginBottom: 8 }}>📁</div><div style={{ fontSize: 11 }}>No images imported</div></div>
            ) : (
              images.map((img, i) => {
                const imgStatus = getImageStatus(regions[img.id]);
                const statusInfo = IMAGE_STATUS[imgStatus];
                return (
                  <div key={img.id} onClick={() => { setCurrentImageIndex(i); setSelectedRegions([]); setZoom(1); setActiveFilter(null); }} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px", marginBottom: 4, borderRadius: 6, cursor: "pointer", background: i === currentImageIndex ? PRIMARY_LIGHT : "#FFF", border: `1px solid ${i === currentImageIndex ? PRIMARY_BORDER : "#E2E8F0"}` }}>
                    {/* Status indicator (NEW) */}
                    <span style={{ color: statusInfo.color, fontSize: 10 }} title={statusInfo.label}>{statusInfo.icon}</span>
                    <div style={{ width: 32, height: 32, borderRadius: 4, overflow: "hidden", background: "#E2E8F0", flexShrink: 0 }}><img src={img.url} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} /></div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 10, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{img.name}</div>
                      <div style={{ fontSize: 9, color: "#A0AEC0" }}>{(regions[img.id] || []).length} obj</div>
                    </div>
                    <button onClick={(e) => handleDeleteImage(img.id, e)} title="Remove" style={{ padding: "2px 6px", fontSize: 12, background: "transparent", border: "none", color: "#A0AEC0", cursor: "pointer", borderRadius: 4 }} onMouseEnter={(e) => { e.target.style.background = "#FED7D7"; e.target.style.color = "#E53E3E"; }} onMouseLeave={(e) => { e.target.style.background = "transparent"; e.target.style.color = "#A0AEC0"; }}>×</button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* CENTER - Canvas */}
        <div style={s.canvas}>
          <div style={s.canvasToolbar}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <TooltipButton onClick={() => currentImageIndex > 0 && setCurrentImageIndex((i) => i - 1)} style={{ padding: "4px 8px", background: "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer" }} title="Previous image (←)" shortcut="←">◀</TooltipButton>
              <span style={{ fontSize: 12, color: "#E2E8F0", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{currentImage?.name || "No image"}</span>
              <span style={{ fontSize: 11, color: "#718096" }}>{currentImageIndex + 1}/{Math.max(1, images.length)}</span>
              <TooltipButton onClick={() => currentImageIndex < images.length - 1 && setCurrentImageIndex((i) => i + 1)} style={{ padding: "4px 8px", background: "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer" }} title="Next image (→)" shortcut="→">▶</TooltipButton>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {Object.entries(ANNOTATION_MODES).map(([key, mode]) => (<TooltipButton key={key} onClick={() => setAnnotationMode(key)} title={`${mode.label} mode`} style={s.modeButton(annotationMode === key)} shortcut={mode.shortcut}><span>{mode.icon}</span><span>{mode.label}</span></TooltipButton>))}
              <div style={{ width: 1, height: 20, background: "#4A5568", margin: "0 8px" }} />
              <button onClick={handleZoomOut} style={{ padding: "4px 8px", background: "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer" }}>−</button>
              <span style={{ fontSize: 11, color: "#E2E8F0", minWidth: 40, textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
              <button onClick={handleZoomIn} style={{ padding: "4px 8px", background: "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer" }}>+</button>
              <button onClick={handleZoomReset} style={{ padding: "4px 8px", fontSize: 10, background: "#4A5568", color: "#FFF", border: "none", borderRadius: 4, cursor: "pointer" }}>Reset</button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <TooltipButton onClick={handleUndo} disabled={historyIndex <= 0} title="Undo (Ctrl+Z)" style={{ padding: "4px 8px", background: historyIndex > 0 ? "#4A5568" : "#2D3748", color: historyIndex > 0 ? "#FFF" : "#718096", border: "none", borderRadius: 4, cursor: historyIndex > 0 ? "pointer" : "not-allowed", fontSize: 12 }}>↩</TooltipButton>
              <TooltipButton onClick={handleRedo} disabled={historyIndex >= history.length - 1} title="Redo (Ctrl+Shift+Z)" style={{ padding: "4px 8px", background: historyIndex < history.length - 1 ? "#4A5568" : "#2D3748", color: historyIndex < history.length - 1 ? "#FFF" : "#718096", border: "none", borderRadius: 4, cursor: historyIndex < history.length - 1 ? "pointer" : "not-allowed", fontSize: 12 }}>↪</TooltipButton>
              <span style={{ fontSize: 11, color: "#718096", marginLeft: 8 }}>{currentRegions.length} obj</span>
              {currentRegions.length > 0 && (<TooltipButton onClick={handleClearCurrentDetections} title="Clear all (with confirm)" style={{ padding: "4px 8px", fontSize: 11, background: "transparent", color: "#A0AEC0", border: "1px solid #4A5568", borderRadius: 4, cursor: "pointer" }}>🗑</TooltipButton>)}
            </div>
          </div>

          <div style={s.canvasArea}>
            {currentImage ? (
              <div ref={canvasRef} onMouseDown={handleCanvasMouseDown} onMouseMove={handleCanvasMouseMove} onMouseUp={handleCanvasMouseUp} onMouseLeave={handleCanvasMouseUp} style={{ width: CANVAS_WIDTH, height: CANVAS_HEIGHT, position: "relative", overflow: "hidden", borderRadius: 8, transform: `scale(${zoom})`, transformOrigin: "center", transition: isDrawing || isDragging || isResizing ? "none" : "transform 0.15s", cursor: getCursor(), userSelect: "none" }}>
                <img src={currentImage.url} alt="" draggable={false} style={{ width: "100%", height: "100%", objectFit: "contain", pointerEvents: "none" }} />
                {currentRegions.map((region, idx) => {
                  const isSelected = selectedRegions.includes(region.id);
                  const isHovered = hoveredRegion === region.id;
                  const scaled = getScaledBoundingBox(region, currentImage.width, currentImage.height);
                  const color = getClassColor(region.label);
                  const showHandles = isSelected && annotationMode === "edit";
                  const showLabel = shouldShowLabel(region.id);
                  const isDimmed = isRegionDimmed(region);
                  const isLowConfidence = region.confidence < LOW_CONFIDENCE_THRESHOLD;
                  
                  return (
                    <div key={region.id} onClick={(e) => handleSelectRegion(region.id, e, idx)} onMouseDown={(e) => { if (annotationMode === "edit" && !isResizing) handleDragStart(region.id, e); }} onDoubleClick={(e) => handleDoubleClick(region.id, e)} onMouseEnter={() => setHoveredRegion(region.id)} onMouseLeave={() => setHoveredRegion(null)} style={{ position: "absolute", left: scaled.x, top: scaled.y, width: scaled.w, height: scaled.h, border: `${isSelected ? 3 : 2}px ${isLowConfidence ? "dashed" : "solid"} ${color}`, background: isSelected ? `${color}30` : isHovered ? `${color}15` : "transparent", borderRadius: 2, cursor: annotationMode === "edit" ? "move" : annotationMode === "select" ? "pointer" : "default", boxSizing: "border-box", opacity: isDimmed ? 0.25 : (isLowConfidence && !isSelected && !isHovered ? 0.7 : 1), transition: "opacity 0.2s" }}>
                      {showLabel && !isDimmed && (<div style={{ position: "absolute", top: -22, left: -2, padding: "2px 6px", background: color, borderRadius: 3, fontSize: 10, fontWeight: 600, color: "#FFF", whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 4, opacity: isLowConfidence ? 0.8 : 1 }}>{region.status === "manual" && <span style={{ opacity: 0.8 }}>✎</span>}{region.label}{showConfidence && <span style={{ opacity: 0.8 }}>{(region.confidence * 100).toFixed(0)}%</span>}</div>)}
                      {showHandles && <ResizeHandles box={scaled} color={color} onResizeStart={(handle, e) => handleResizeStart(region.id, handle, e)} />}
                    </div>
                  );
                })}
                {isDrawing && drawStart && drawCurrent && (<div style={{ position: "absolute", left: Math.min(drawStart.x, drawCurrent.x), top: Math.min(drawStart.y, drawCurrent.y), width: Math.abs(drawCurrent.x - drawStart.x), height: Math.abs(drawCurrent.y - drawStart.y), border: `2px dashed ${PRIMARY_COLOR}`, background: `${PRIMARY_COLOR}20`, borderRadius: 2, pointerEvents: "none" }} />)}
              </div>
            ) : (
              <div style={{ textAlign: "center", color: "#718096" }}><div style={{ fontSize: 48, marginBottom: 16 }}>📷</div><div style={{ marginBottom: 12 }}>No image selected</div><button onClick={handleImportClick} style={{ padding: "10px 20px", background: PRIMARY_COLOR, color: "#FFF", border: "none", borderRadius: 8, cursor: "pointer" }}>Import Images</button></div>
            )}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={s.rightPanel}>
          <div style={{ padding: "12px", borderBottom: "1px solid #E2E8F0" }}>
            <div style={s.sectionTitle}>Sensitivity</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 10 }}>
              {Object.entries(SENSITIVITY_PRESETS).map(([key, p]) => (<button key={key} onClick={() => applySensitivityPreset(key)} style={{ flex: 1, padding: "6px", fontSize: 10, textAlign: "center", border: `1px solid ${sensitivityPreset === key ? PRIMARY_COLOR : "#E2E8F0"}`, borderRadius: 4, background: sensitivityPreset === key ? PRIMARY_LIGHT : "#FFF", cursor: "pointer" }}><div style={{ fontWeight: 600 }}>{p.label}</div></button>))}
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={s.sectionTitle}>Thresholds</span>
              <button onClick={() => setShowThresholds(!showThresholds)} style={{ fontSize: 10, color: "#718096", background: "none", border: "none", cursor: "pointer" }}>{showThresholds ? "Hide" : "Show"}</button>
            </div>
            {showThresholds && currentModelConfig?.thresholds && (<div style={{ marginBottom: 8 }}>{Object.entries(currentModelConfig.thresholds).map(([k, v]) => (<div key={k} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}><span style={{ fontSize: 10, color: "#718096", minWidth: 50 }}>{v.label}</span><input type="range" min={v.min} max={v.max} step={0.05} value={thresholds[k] || v.default} onChange={(e) => { setThresholds((p) => ({ ...p, [k]: parseFloat(e.target.value) })); setSensitivityPreset(null); }} style={{ flex: 1, cursor: "pointer" }} /><span style={{ fontSize: 10, fontWeight: 600, minWidth: 28 }}>{((thresholds[k] || v.default) * 100).toFixed(0)}%</span></div>))}</div>)}
            <div style={{ marginTop: 10 }}>
              <div style={s.sectionTitle}>Label Display</div>
              <div style={{ display: "flex", gap: 4 }}>{Object.entries(LABEL_DISPLAY_MODES).map(([key, mode]) => (<button key={key} onClick={() => setLabelDisplayMode(key)} title={mode.desc} style={{ flex: 1, padding: "4px 6px", fontSize: 10, border: `1px solid ${labelDisplayMode === key ? PRIMARY_COLOR : "#E2E8F0"}`, borderRadius: 4, background: labelDisplayMode === key ? PRIMARY_LIGHT : "#FFF", cursor: "pointer" }}>{mode.label}</button>))}</div>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#718096", cursor: "pointer", marginTop: 10 }}><input type="checkbox" checked={showConfidence} onChange={(e) => setShowConfidence(e.target.checked)} />Show confidence</label>
          </div>

          <div style={{ padding: "10px 12px", borderBottom: "1px solid #E2E8F0" }}>
            <div style={s.sectionTitle}>Filter by Class</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {filterChips.map((chip) => (<FilterChip key={chip.label} label={chip.label} count={chip.count} color={chip.color} isActive={chip.label === "All" ? !activeFilter : activeFilter === chip.label} onClick={() => handleFilterClick(chip.label)} />))}
            </div>
          </div>

          <div style={{ padding: "10px 12px", borderBottom: "1px solid #E2E8F0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 600 }}>Labeled Objects</span>
            <span style={{ padding: "2px 8px", background: "#EDF2F7", borderRadius: 10, fontSize: 11 }}>{currentRegions.length}</span>
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: 8 }}>
            {currentRegions.length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "#A0AEC0", fontSize: 11 }}>{annotationMode === "draw" ? (<><div style={{ fontSize: 24, marginBottom: 8 }}>✎</div><div>Draw boxes to annotate</div></>) : "No objects detected"}</div>
            ) : (
              groupedRegions.filter(([label]) => !activeFilter || label === activeFilter).map(([label, items]) => (
                <CategoryAccordion key={label} label={label} color={getClassColor(label)} regions={items} isExpanded={expandedCategories.has(label)} onToggle={() => toggleCategory(label)} selectedRegions={selectedRegions} onSelectRegion={handleSelectRegion} onDoubleClick={handleDoubleClick} hoveredRegion={hoveredRegion} setHoveredRegion={setHoveredRegion} activeFilter={activeFilter} lastSelectedIndex={lastSelectedIndex} setLastSelectedIndex={setLastSelectedIndex} allRegionIds={allRegionIds} />
              ))
            )}
          </div>

          <div style={{ padding: "12px", borderTop: "1px solid #E2E8F0", background: "#F7FAFC" }}>
            <div style={{ fontSize: 11, color: "#718096", marginBottom: 8 }}>{selectedRegions.length > 0 ? `${selectedRegions.length} selected • Shift+Click for range` : "Select objects to review"}</div>
            <div style={{ display: "flex", gap: 6 }}>
              <TooltipButton onClick={handleApprove} disabled={!selectedRegions.length} title="Approve selected (Enter)" style={{ flex: 1, padding: "6px", fontSize: 11, background: selectedRegions.length ? "#38A169" : "#E2E8F0", color: selectedRegions.length ? "#FFF" : "#A0AEC0", border: "none", borderRadius: 4, cursor: selectedRegions.length ? "pointer" : "not-allowed" }} shortcut="↵">✓ Approve</TooltipButton>
              <TooltipButton onClick={handleFlag} disabled={!selectedRegions.length} title="Flag for review" style={{ flex: 1, padding: "6px", fontSize: 11, background: selectedRegions.length ? "#FFF" : "#E2E8F0", color: selectedRegions.length ? "#D69E2E" : "#A0AEC0", border: "1px solid", borderColor: selectedRegions.length ? "#D69E2E" : "#E2E8F0", borderRadius: 4, cursor: selectedRegions.length ? "pointer" : "not-allowed" }}>⚠ Flag</TooltipButton>
              <TooltipButton onClick={handleDelete} disabled={!selectedRegions.length} title="Delete selected (Del)" style={{ flex: 1, padding: "6px", fontSize: 11, background: selectedRegions.length ? "#FFF" : "#E2E8F0", color: selectedRegions.length ? "#E53E3E" : "#A0AEC0", border: "1px solid", borderColor: selectedRegions.length ? "#E53E3E" : "#E2E8F0", borderRadius: 4, cursor: selectedRegions.length ? "pointer" : "not-allowed" }}>🗑</TooltipButton>
            </div>
          </div>

          <div style={{ padding: "12px", borderTop: "1px solid #E2E8F0" }}>
            <button onClick={() => setShowExportModal(true)} disabled={totalRegionsAllImages === 0} style={{ width: "100%", padding: "10px", fontSize: 13, fontWeight: 600, background: totalRegionsAllImages === 0 ? "#E2E8F0" : PRIMARY_COLOR, color: totalRegionsAllImages === 0 ? "#A0AEC0" : "#FFF", border: "none", borderRadius: 6, cursor: totalRegionsAllImages === 0 ? "not-allowed" : "pointer" }}>Export ({totalRegionsAllImages} labels)</button>
          </div>
        </div>
      </div>

      <div style={{ padding: "6px 20px", background: "#FFF", borderTop: "1px solid #E2E8F0", fontSize: 10, color: "#718096", display: "flex", justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 12 }}><span>V Select</span><span>D Draw</span><span>E Edit</span><span>← → Nav</span><span>Del Delete</span><span>Enter Approve</span><span>Shift+Click Range</span><span>Right-click Menu</span></div>
        <span>{currentModelConfig?.name} @ {currentServerUrl}</span>
      </div>
    </div>
  );
}
