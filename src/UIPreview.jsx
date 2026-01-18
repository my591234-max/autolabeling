import React, { useState } from 'react';

/**
 * AutoLabel Pro - UI Redesign Preview
 * Professional academic labeling tool with scalable model selector
 */

const MODEL_CONFIGS = {
  "yolo-v8": {
    id: "yolo-v8",
    name: "YOLOv8",
    category: "Closed-Set Detection",
    description: "Fast object detection with 80 COCO classes",
    requiresPrompt: false,
  },
  "grounding-dino": {
    id: "grounding-dino",
    name: "Grounding DINO",
    category: "Open-World Detection",
    description: "Text-guided detection - describe what to find",
    requiresPrompt: true,
  },
  "yolo-world": {
    id: "yolo-world",
    name: "YOLO-World",
    category: "Open-World Detection",
    description: "Real-time open-vocabulary detection",
    requiresPrompt: true,
    disabled: true,
  },
  "owl-vit": {
    id: "owl-vit",
    name: "OWL-ViT",
    category: "Open-World Detection",
    description: "Vision Transformer for open-vocabulary detection",
    requiresPrompt: true,
    disabled: true,
  },
};

export default function AutoLabelProPreview() {
  const [selectedModelId, setSelectedModelId] = useState("grounding-dino");
  const [textPrompt, setTextPrompt] = useState("car . person . bicycle");
  const [boxThreshold, setBoxThreshold] = useState(0.25);
  const [textThreshold, setTextThreshold] = useState(0.20);
  const [isConnected, setIsConnected] = useState(true);
  
  const currentModel = MODEL_CONFIGS[selectedModelId];
  const requiresPrompt = currentModel?.requiresPrompt;

  return (
    <div style={{
      fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
      background: '#f7fafc',
      minHeight: '100vh',
      color: '#2d3748'
    }}>
      {/* ===== Header ===== */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 16px',
        background: 'linear-gradient(to bottom, #fff, #fafbfc)',
        borderBottom: '1px solid #e2e8f0',
        boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 32,
              height: 32,
              background: 'linear-gradient(135deg, #2b6cb0, #1a365d)',
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontWeight: 700,
              fontSize: 12,
              boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
            }}>AL</div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>
              Auto<span style={{ color: '#2b6cb0' }}>Label</span> Pro
            </div>
            <span style={{
              fontSize: 10,
              padding: '2px 6px',
              background: '#edf2f7',
              color: '#718096',
              borderRadius: 10,
              fontWeight: 500
            }}>v2.0</span>
          </div>
          <div style={{ 
            fontSize: 12, 
            color: '#718096', 
            borderLeft: '1px solid #e2e8f0', 
            paddingLeft: 12 
          }}>
            逢甲大學自動駕駛研究團隊
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: 8 }}>
          <button style={{
            padding: '6px 12px',
            fontSize: 13,
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 6,
            cursor: 'pointer'
          }}>📁 Import Images</button>
          <button style={{
            padding: '6px 12px',
            fontSize: 13,
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 6,
            cursor: 'pointer'
          }}>⚙️ Model Setup</button>
          <button style={{
            padding: '6px 12px',
            fontSize: 13,
            background: 'linear-gradient(to bottom, #3182ce, #2b6cb0)',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
          }}>📤 Export (24)</button>
        </div>
      </header>

      {/* ===== Toolbar ===== */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '8px 16px',
        background: '#fff',
        borderBottom: '1px solid #e2e8f0',
        flexWrap: 'wrap'
      }}>
        {/* Tool Buttons */}
        <div style={{ display: 'flex', gap: 4 }}>
          <button style={{
            padding: '5px 10px',
            fontSize: 12,
            background: '#fff',
            border: '1px solid #e2e8f0',
            borderRadius: 4,
            cursor: 'pointer'
          }}>👆 Select</button>
          <button style={{
            padding: '5px 10px',
            fontSize: 12,
            background: 'linear-gradient(to bottom, #3182ce, #2b6cb0)',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer'
          }}>▢ Rectangle</button>
        </div>

        <div style={{ width: 1, height: 24, background: '#e2e8f0' }} />

        {/* Model Selector Panel */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '8px 12px',
          background: '#f7fafc',
          border: '1px solid #e2e8f0',
          borderRadius: 8
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              color: '#a0aec0',
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}>Detection Model</span>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              style={{
                padding: '6px 32px 6px 10px',
                fontSize: 13,
                fontWeight: 500,
                border: '1px solid #cbd5e0',
                borderRadius: 4,
                background: '#fff',
                cursor: 'pointer',
                minWidth: 180
              }}
            >
              <optgroup label="Closed-Set Detection">
                <option value="yolo-v8">YOLOv8</option>
              </optgroup>
              <optgroup label="Open-World Detection (Prompt-Guided)">
                <option value="grounding-dino">Grounding DINO</option>
                <option value="yolo-world" disabled>YOLO-World (Coming Soon)</option>
                <option value="owl-vit" disabled>OWL-ViT (Coming Soon)</option>
              </optgroup>
            </select>
          </div>
          
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            background: isConnected ? '#f0fff4' : '#f7fafc',
            border: `1px solid ${isConnected ? '#48bb78' : '#cbd5e0'}`,
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 500,
            color: isConnected ? '#2f855a' : '#718096'
          }}>
            <span style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: isConnected ? '#48bb78' : '#a0aec0'
            }} />
            {isConnected ? 'Ready' : 'Not Connected'}
          </div>
        </div>

        {/* Prompt Panel (for Open-World Models) */}
        {requiresPrompt && (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            minWidth: 280,
            maxWidth: 450
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{
                fontSize: 10,
                fontWeight: 600,
                color: '#a0aec0',
                textTransform: 'uppercase',
                letterSpacing: 0.5
              }}>Detection Prompt</span>
              <span style={{
                fontSize: 10,
                color: '#a0aec0',
                fontFamily: "'IBM Plex Mono', monospace"
              }}>Separate with periods: car . person</span>
            </div>
            <textarea
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              placeholder="Describe objects to detect...&#10;&#10;Examples:&#10;• person . car . bicycle&#10;• red car . pedestrian"
              style={{
                width: '100%',
                minHeight: 60,
                padding: '8px 12px',
                fontSize: 13,
                lineHeight: 1.4,
                border: '1px solid #cbd5e0',
                borderRadius: 6,
                resize: 'vertical',
                fontFamily: 'inherit'
              }}
            />
          </div>
        )}

        {/* Threshold Controls */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12 }}>
          {requiresPrompt ? (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 10, fontWeight: 500, color: '#718096' }}>Box Threshold</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={boxThreshold}
                    onChange={(e) => setBoxThreshold(parseFloat(e.target.value))}
                    style={{ width: 70 }}
                  />
                  <span style={{
                    fontSize: 11,
                    fontWeight: 600,
                    fontFamily: "'IBM Plex Mono', monospace",
                    color: '#4a5568',
                    minWidth: 32
                  }}>{(boxThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 10, fontWeight: 500, color: '#718096' }}>Text Threshold</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={textThreshold}
                    onChange={(e) => setTextThreshold(parseFloat(e.target.value))}
                    style={{ width: 70 }}
                  />
                  <span style={{
                    fontSize: 11,
                    fontWeight: 600,
                    fontFamily: "'IBM Plex Mono', monospace",
                    color: '#4a5568',
                    minWidth: 32
                  }}>{(textThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 10, fontWeight: 500, color: '#718096' }}>Confidence</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input type="range" style={{ width: 70 }} />
                  <span style={{ fontSize: 11, fontFamily: 'monospace' }}>25%</span>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontSize: 10, fontWeight: 500, color: '#718096' }}>IoU (NMS)</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input type="range" style={{ width: 70 }} />
                  <span style={{ fontSize: 11, fontFamily: 'monospace' }}>45%</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Detect Button */}
        <button style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '10px 20px',
          fontSize: 13,
          fontWeight: 600,
          background: 'linear-gradient(135deg, #f59e0b, #d97706)',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          cursor: 'pointer',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.2)'
        }}>
          🔍 Detect
        </button>
        
        <button style={{
          padding: '5px 10px',
          fontSize: 12,
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: 4,
          cursor: 'pointer'
        }}>All (5)</button>

        <div style={{ width: 1, height: 24, background: '#e2e8f0' }} />

        <button style={{
          padding: '5px 10px',
          fontSize: 12,
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: 4,
          cursor: 'pointer'
        }}>🏷️ Labels</button>
      </div>

      {/* ===== Main Content ===== */}
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        height: 'calc(100vh - 130px)'
      }}>
        {/* Left Panel - Image List */}
        <div style={{
          width: 220,
          background: '#fff',
          borderRight: '1px solid #e2e8f0',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 16px',
            borderBottom: '1px solid #e2e8f0',
            background: '#f7fafc'
          }}>
            <span style={{
              fontSize: 12,
              fontWeight: 600,
              color: '#4a5568',
              textTransform: 'uppercase',
              letterSpacing: 0.5
            }}>Images</span>
            <span style={{
              fontSize: 11,
              padding: '2px 8px',
              background: '#edf2f7',
              color: '#718096',
              borderRadius: 10
            }}>5</span>
          </div>
          <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
            {['img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_004.jpg', 'img_005.jpg'].map((name, i) => (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: 8,
                marginBottom: 4,
                borderRadius: 6,
                cursor: 'pointer',
                background: i === 0 ? '#ebf8ff' : 'transparent',
                border: i === 0 ? '1px solid #4299e1' : '1px solid transparent'
              }}>
                <div style={{
                  width: 48,
                  height: 48,
                  background: '#e2e8f0',
                  borderRadius: 4,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 20
                }}>🖼️</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 500,
                    color: '#2d3748',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>{name}</div>
                  <div style={{ fontSize: 10, color: '#a0aec0', marginTop: 2 }}>
                    {i === 0 && <span style={{
                      padding: '1px 6px',
                      background: '#f0fff4',
                      color: '#2f855a',
                      borderRadius: 8,
                      fontSize: 10
                    }}>✓ 3 labels</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Center Panel - Canvas */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          background: '#edf2f7'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 16px',
            background: '#fff',
            borderBottom: '1px solid #e2e8f0',
            fontSize: 12,
            color: '#718096'
          }}>
            <span>img_001.jpg • 1920 × 1080</span>
            <span>Zoom: 100%</span>
          </div>
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `
              linear-gradient(45deg, #cbd5e0 25%, transparent 25%),
              linear-gradient(-45deg, #cbd5e0 25%, transparent 25%),
              linear-gradient(45deg, transparent 75%, #cbd5e0 75%),
              linear-gradient(-45deg, transparent 75%, #cbd5e0 75%)
            `,
            backgroundSize: '20px 20px',
            backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px'
          }}>
            <div style={{
              width: 600,
              height: 340,
              background: '#718096',
              borderRadius: 4,
              boxShadow: '0 10px 15px rgba(0,0,0,0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: 48
            }}>
              🚗
            </div>
          </div>
        </div>

        {/* Right Panel - Regions */}
        <div style={{
          width: 280,
          background: '#fff',
          borderLeft: '1px solid #e2e8f0',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{
            display: 'flex',
            borderBottom: '1px solid #e2e8f0',
            background: '#f7fafc'
          }}>
            {['Regions', 'Labels', 'Help'].map((tab, i) => (
              <button key={tab} style={{
                flex: 1,
                padding: 12,
                fontSize: 12,
                fontWeight: 500,
                border: 'none',
                cursor: 'pointer',
                background: i === 0 ? '#fff' : 'transparent',
                color: i === 0 ? '#2b6cb0' : '#718096',
                borderBottom: i === 0 ? '2px solid #3182ce' : '2px solid transparent'
              }}>{tab}</button>
            ))}
          </div>
          <div style={{ flex: 1, overflow: 'auto', padding: 8 }}>
            {[
              { label: 'car', color: '#e53e3e', conf: 0.92, ai: true },
              { label: 'person', color: '#38a169', conf: 0.87, ai: true },
              { label: 'bicycle', color: '#3182ce', conf: 0.78, ai: true },
            ].map((region, i) => (
              <div key={i} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 12px',
                marginBottom: 4,
                borderRadius: 6,
                background: i === 0 ? '#ebf8ff' : '#f7fafc',
                border: `1px solid ${i === 0 ? '#4299e1' : '#e2e8f0'}`,
                cursor: 'pointer'
              }}>
                <div style={{
                  width: 12,
                  height: 12,
                  borderRadius: 3,
                  background: region.color
                }} />
                <span style={{
                  flex: 1,
                  fontSize: 12,
                  fontWeight: 500,
                  color: '#2d3748'
                }}>{region.label}</span>
                <span style={{
                  fontSize: 10,
                  fontFamily: "'IBM Plex Mono', monospace",
                  color: '#a0aec0',
                  padding: '2px 6px',
                  background: '#edf2f7',
                  borderRadius: 4
                }}>{(region.conf * 100).toFixed(0)}%</span>
                {region.ai && <span style={{
                  fontSize: 9,
                  fontWeight: 600,
                  padding: '2px 5px',
                  background: '#fffbeb',
                  color: '#d97706',
                  borderRadius: 3
                }}>AI</span>}
              </div>
            ))}
          </div>
          
          {/* Label Selector */}
          <div style={{
            padding: '12px 16px',
            background: '#f7fafc',
            borderTop: '1px solid #e2e8f0'
          }}>
            <div style={{
              fontSize: 10,
              fontWeight: 600,
              color: '#a0aec0',
              textTransform: 'uppercase',
              letterSpacing: 0.5,
              marginBottom: 8
            }}>Quick Labels</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {[
                { name: 'car', color: '#e53e3e', key: '1' },
                { name: 'person', color: '#38a169', key: '2' },
                { name: 'bicycle', color: '#3182ce', key: '3' },
                { name: 'truck', color: '#d69e2e', key: '4' },
              ].map((label) => (
                <button key={label.name} style={{
                  padding: '5px 10px',
                  fontSize: 11,
                  fontWeight: 500,
                  color: '#fff',
                  background: label.color,
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  opacity: 0.7
                }}>
                  {label.name} <span style={{ opacity: 0.7 }}>({label.key})</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ===== Design Notes ===== */}
      <div style={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        padding: 16,
        background: '#1a202c',
        color: '#fff',
        borderRadius: 8,
        fontSize: 12,
        maxWidth: 300,
        boxShadow: '0 10px 15px rgba(0,0,0,0.2)'
      }}>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>🎨 UI Redesign Highlights</div>
        <ul style={{ margin: 0, paddingLeft: 16, lineHeight: 1.6 }}>
          <li><strong>Dropdown Model Selector</strong> - Scalable for future models</li>
          <li><strong>Expanded Prompt Area</strong> - Multi-line textarea</li>
          <li><strong>Academic Color Palette</strong> - Professional blue theme</li>
          <li><strong>Clear Visual Hierarchy</strong> - Grouped controls</li>
          <li><strong>Keyboard Hints</strong> - Ctrl+Enter to detect</li>
        </ul>
      </div>
    </div>
  );
}
