/**
 * COCO 資料集的 80 個類別名稱
 */
export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane',
  'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
  'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
  'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
  'wine glass', 'cup', 'fork', 'knife', 'spoon',
  'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
  'cake', 'chair', 'couch', 'potted plant', 'bed',
  'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
  'toaster', 'sink', 'refrigerator', 'book', 'clock',
  'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

/**
 * 預設標籤設定
 */
export const DEFAULT_LABELS = [
  { id: 1, name: 'car', color: '#FF6B6B', hotkey: '1' },
  { id: 2, name: 'person', color: '#4ECDC4', hotkey: '2' },
  { id: 3, name: 'motorcycle', color: '#45B7D1', hotkey: '3' },
  { id: 4, name: 'bicycle', color: '#96CEB4', hotkey: '4' },
  { id: 5, name: 'truck', color: '#FFEAA7', hotkey: '5' },
  { id: 6, name: 'bus', color: '#DDA0DD', hotkey: '6' },
];

/**
 * 預設自動標註類別
 */
export const DEFAULT_AUTO_LABEL_CLASSES = [
  'car', 'person', 'motorcycle', 'bicycle', 'truck', 'bus'
];

/**
 * 匯出格式選項
 */
export const EXPORT_FORMATS = [
  { value: 'yolo', label: 'YOLO', description: 'Normalized .txt files' },
  { value: 'coco', label: 'COCO JSON', description: 'Single JSON file' }
];

/**
 * 可選的類別過濾器
 */
export const FILTER_CLASSES = [
  'car', 'person', 'motorcycle', 'bicycle', 'truck', 'bus', 'traffic light', 'stop sign'
];
