// Web Body Tracker - Client-side MediaPipe (Zero Latency)
const video = document.getElementById('video');

// CRITICAL FIX: Use TWO separate canvases to prevent flicker!
// Canvas 1: Overlay canvas for video + hand tracking (redrawn every frame)
const canvasOverlay = document.getElementById('canvasOverlay');
const ctxOverlay = canvasOverlay.getContext('2d', {
    alpha: true,
    desynchronized: false  // FIXED: Changed to false to prevent black screen on modern GPUs
});

// Canvas 2: Painting canvas for persistent drawing (NEVER cleared during tracking)
const canvasPainting = document.getElementById('canvasPainting');
const ctxPainting = canvasPainting.getContext('2d', {
    alpha: true,
    desynchronized: false,  // FIXED: Changed to false to prevent black screen on modern GPUs
    willReadFrequently: true  // PERFORMANCE FIX: Optimize for getImageData operations
});

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const videoContainer = document.querySelector('.video-container');
const editPaintingBtn = document.getElementById('editPaintingBtn');
const saveBtn = document.getElementById('saveBtn');
const colorPickerBtn = document.getElementById('colorPickerBtn');
const colorPickerModal = document.getElementById('colorPickerModal');
const colorWheelCanvas = document.getElementById('colorWheelCanvas');
const colorSelectorDot = document.getElementById('colorSelectorDot');
const brightnessSlider = document.getElementById('brightnessSlider');
const brightnessHandle = document.getElementById('brightnessHandle');
const donePickerBtn = document.getElementById('donePickerBtn');
const fullscreenToggleBtn = document.getElementById('fullscreenToggleBtn');
const fullscreenIcon = document.getElementById('fullscreenIcon');
const paintBtn = document.getElementById('paintBtn');
const eraseBtn = document.getElementById('eraseBtn');
const brushIndicator = document.getElementById('brushIndicator');
const brushSizeSliderContainer = document.getElementById('brushSizeSliderContainer');
const brushSizeSlider = document.getElementById('brushSizeSlider');
const brushSizeHandle = document.getElementById('brushSizeHandle');
const brushSizeValue = document.getElementById('brushSizeValue');

let camera = null;
let hands = null;
let isTracking = false;
let savedPaintingData = null; // Store painting without white background
let currentColor = '#FFFFFF'; // Start with white
let isPaintMode = true; // Start in paint mode
let currentHue = 0; // 0-360 (doesn't matter when saturation is 0)
let currentSaturation = 0; // 0-100 (0 = white/gray, no color)
let currentBrightness = 100; // 0-100 (100 = white when sat=0)
let isColorPickerOpen = false;

// Button hover detection
const fullscreenProgressCircle = fullscreenToggleBtn.querySelector('.progress-ring circle');
const colorProgressCircle = colorPickerBtn.querySelector('.progress-ring circle');
const doneProgressRect = donePickerBtn.querySelector('.progress-ring rect');
const paintProgressCircle = paintBtn.querySelector('.progress-ring circle');
const eraseProgressCircle = eraseBtn.querySelector('.progress-ring circle');
let fullscreenHoverStartTime = null;
let fullscreenHoverProgress = 0;
let colorHoverStartTime = null;
let colorHoverProgress = 0;
let doneHoverStartTime = null;
let doneHoverProgress = 0;
let paintHoverStartTime = null;
let paintHoverProgress = 0;
let eraseHoverStartTime = null;
let eraseHoverProgress = 0;
const HOVER_DURATION = 1000; // 1 second to activate

// Color wheel state
let colorWheelCtx = null;
let selectorDotX = 200; // Center of wheel
let selectorDotY = 200; // Center of wheel

// Painting variables
let lastPaintPoints = {}; // Store last point for each finger independently
let fingerLastSeen = {}; // Track when each finger was last seen
let isPainting = false;
let showPaintingOnly = false;
const FINGER_TIMEOUT = 200; // ms - how long to wait before clearing a finger's last point
const MAX_LINE_DISTANCE = 100; // pixels - max distance for a valid line (prevents jumps)
let needsAlphaCap = false; // Flag to cap alpha when fingers are lifted
let lastAlphaCapTime = 0; // Throttle alpha capping

// Brush size settings
const MIN_BRUSH_SIZE = 1;
const MAX_BRUSH_SIZE = 50;
let currentBrushSize = 1; // Default brush size

// Draw color wheel
function drawColorWheel() {
    if (!colorWheelCtx) {
        colorWheelCtx = colorWheelCanvas.getContext('2d');
    }

    const centerX = colorWheelCanvas.width / 2;
    const centerY = colorWheelCanvas.height / 2;
    const radius = colorWheelCanvas.width / 2;

    // Draw color wheel
    for (let angle = 0; angle < 360; angle += 1) {
        const startAngle = (angle - 0.5) * Math.PI / 180;
        const endAngle = (angle + 0.5) * Math.PI / 180;

        colorWheelCtx.beginPath();
        colorWheelCtx.moveTo(centerX, centerY);
        colorWheelCtx.arc(centerX, centerY, radius, startAngle, endAngle);
        colorWheelCtx.closePath();

        // Create gradient from white (center) to full color (edge)
        const gradient = colorWheelCtx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
        gradient.addColorStop(0, 'white');
        gradient.addColorStop(1, `hsl(${angle}, 100%, 50%)`);

        colorWheelCtx.fillStyle = gradient;
        colorWheelCtx.fill();
    }
}

// Convert HSB to RGB (HSB = HSV model)
function hsbToRgb(h, s, v) {
    s = s / 100;
    v = v / 100;

    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;

    let r, g, b;
    if (h >= 0 && h < 60) {
        [r, g, b] = [c, x, 0];
    } else if (h >= 60 && h < 120) {
        [r, g, b] = [x, c, 0];
    } else if (h >= 120 && h < 180) {
        [r, g, b] = [0, c, x];
    } else if (h >= 180 && h < 240) {
        [r, g, b] = [0, x, c];
    } else if (h >= 240 && h < 300) {
        [r, g, b] = [x, 0, c];
    } else {
        [r, g, b] = [c, 0, x];
    }

    return [
        Math.round((r + m) * 255),
        Math.round((g + m) * 255),
        Math.round((b + m) * 255)
    ];
}

// Convert RGB to Hex
function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// Update current color from HSB values
function updateColorFromHSB() {
    const [r, g, b] = hsbToRgb(currentHue, currentSaturation, currentBrightness);
    currentColor = rgbToHex(r, g, b);
    colorPickerBtn.style.backgroundColor = currentColor;
}

// Initialize color picker
function initColorPicker() {
    drawColorWheel();
    updateColorFromHSB(); // Make sure currentColor matches HSB values
    updateBrightnessSlider();
    updateSelectorDotPosition();
}

// Update selector dot position based on hue and saturation
function updateSelectorDotPosition() {
    // Get actual color wheel dimensions
    const containerRect = document.querySelector('.color-wheel-container').getBoundingClientRect();

    const centerX = containerRect.width / 2;
    const centerY = containerRect.height / 2;
    const radius = containerRect.width / 2;

    // Convert hue to angle in radians (hue 0° = right, 90° = down, matching canvas arc)
    const angleRad = currentHue * Math.PI / 180;
    const distance = (currentSaturation / 100) * radius;

    // Calculate position
    const dotX = centerX + distance * Math.cos(angleRad);
    const dotY = centerY + distance * Math.sin(angleRad);

    colorSelectorDot.style.left = dotX + 'px';
    colorSelectorDot.style.top = dotY + 'px';
    colorSelectorDot.style.backgroundColor = currentColor;
}

// Update brightness slider handle position
function updateBrightnessSlider() {
    // Get actual slider dimensions
    const sliderRect = brightnessSlider.getBoundingClientRect();
    const sliderHeight = sliderRect.height;
    const position = (1 - currentBrightness / 100) * sliderHeight;
    brightnessHandle.style.top = position + 'px';

    // Update handle color to show current brightness
    const [r, g, b] = hsbToRgb(currentHue, currentSaturation, currentBrightness);
    brightnessHandle.style.backgroundColor = rgbToHex(r, g, b);

    // Update slider background gradient to show current color at full brightness to black
    const [rFull, gFull, bFull] = hsbToRgb(currentHue, currentSaturation, 100);
    const fullBrightnessColor = rgbToHex(rFull, gFull, bFull);
    brightnessSlider.style.background = `linear-gradient(to bottom, ${fullBrightnessColor}, #000000)`;
}

// Handle color wheel interaction (finger pointing)
function handleColorWheelInteraction(x, y) {
    const rect = colorWheelCanvas.getBoundingClientRect();
    const canvasX = x - rect.left;
    const canvasY = y - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const dx = canvasX - centerX;
    const dy = canvasY - centerY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const radius = rect.width / 2;

    // Only update if within the wheel
    if (distance <= radius) {
        // Calculate hue from angle (atan2 returns -180 to 180, we want 0 to 360)
        // atan2(dy, dx) gives us the angle where 0° = right, 90° = down
        let angle = Math.atan2(dy, dx) * 180 / Math.PI;
        // Convert to 0-360 range
        currentHue = (angle + 360) % 360;

        // Calculate saturation from distance
        currentSaturation = Math.min(100, (distance / radius) * 100);

        updateColorFromHSB();
        updateSelectorDotPosition();
        updateBrightnessSlider();
    }
}

// Handle brightness slider interaction (finger pointing)
function handleBrightnessSliderInteraction(x, y) {
    const rect = brightnessSlider.getBoundingClientRect();
    const sliderY = y - rect.top;
    const sliderHeight = rect.height;

    // Clamp to slider bounds
    const clampedY = Math.max(0, Math.min(sliderHeight, sliderY));

    // Calculate brightness (inverted - top is 100%, bottom is 0%)
    currentBrightness = 100 - (clampedY / sliderHeight) * 100;

    updateColorFromHSB();
    updateBrightnessSlider();
    updateSelectorDotPosition();
}

// Open color picker
function openColorPicker() {
    isColorPickerOpen = true;
    colorPickerModal.classList.add('active');

    // Update positions after modal is visible and layout is complete
    setTimeout(() => {
        updateSelectorDotPosition();
        updateBrightnessSlider();
    }, 50);
}

// Close color picker
function closeColorPicker() {
    isColorPickerOpen = false;
    colorPickerModal.classList.remove('active');
    // Reset color button progress
    colorHoverStartTime = null;
    colorHoverProgress = 0;
    colorProgressCircle.style.strokeDashoffset = 170;
    colorPickerBtn.classList.remove('hovering');
    // Reset done button progress
    doneHoverStartTime = null;
    doneHoverProgress = 0;
    // Calculate perimeter dynamically for reset
    const btnRect = donePickerBtn.getBoundingClientRect();
    const perimeter = 2 * (btnRect.width + btnRect.height);
    doneProgressRect.style.strokeDashoffset = perimeter;
    donePickerBtn.classList.remove('hovering');
}

// Switch to paint mode
function setPaintMode() {
    isPaintMode = true;
    paintBtn.classList.add('active');
    eraseBtn.classList.remove('active');
    updateBrushIndicatorStyle();
}

// Switch to erase mode
function setEraseMode() {
    isPaintMode = false;
    paintBtn.classList.remove('active');
    eraseBtn.classList.add('active');
    updateBrushIndicatorStyle();
}

// Store brush indicator elements
const brushIndicators = new Map(); // Map of fingerId -> indicator element

// Create or get brush indicator for a finger
function getBrushIndicator(fingerId) {
    if (!brushIndicators.has(fingerId)) {
        const indicator = document.createElement('div');
        indicator.className = 'brush-indicator';
        indicator.style.position = 'fixed';
        indicator.style.borderRadius = '50%';
        indicator.style.pointerEvents = 'none';
        indicator.style.zIndex = '9999';
        indicator.style.border = '2px solid rgba(255, 255, 255, 0.8)';
        indicator.style.background = 'rgba(255, 255, 255, 0.2)';
        indicator.style.boxShadow = '0 0 10px rgba(0, 0, 0, 0.3)';
        document.body.appendChild(indicator);
        brushIndicators.set(fingerId, indicator);
    }
    return brushIndicators.get(fingerId);
}

// Update brush indicator style based on mode
function updateBrushIndicatorStyle(indicator) {
    if (!indicator) return;

    if (isPaintMode) {
        indicator.classList.remove('erase-mode');
        indicator.classList.add('paint-mode');
        indicator.style.borderColor = 'rgba(76, 175, 80, 0.8)';
        indicator.style.background = 'rgba(76, 175, 80, 0.2)';
    } else {
        indicator.classList.remove('paint-mode');
        indicator.classList.add('erase-mode');
        indicator.style.borderColor = 'rgba(255, 87, 34, 0.8)';
        indicator.style.background = 'rgba(255, 87, 34, 0.2)';
    }
    const size = currentBrushSize * 2;
    indicator.style.width = size + 'px';
    indicator.style.height = size + 'px';
}

// Update all brush indicators
function updateAllBrushIndicators() {
    brushIndicators.forEach(indicator => {
        updateBrushIndicatorStyle(indicator);
    });
}

// Update brush indicator position
function updateBrushIndicatorPosition(fingerId, x, y) {
    if (isNaN(x) || isNaN(y)) return;

    const indicator = getBrushIndicator(fingerId);
    updateBrushIndicatorStyle(indicator);

    const size = currentBrushSize * 2;
    indicator.style.left = (x - size / 2) + 'px';
    indicator.style.top = (y - size / 2) + 'px';
    indicator.style.display = 'block';
}

// Hide specific brush indicator
function hideBrushIndicator(fingerId) {
    const indicator = brushIndicators.get(fingerId);
    if (indicator) {
        indicator.style.display = 'none';
    }
}

// Hide all brush indicators
function hideAllBrushIndicators() {
    brushIndicators.forEach(indicator => {
        indicator.style.display = 'none';
    });
}

// PERFORMANCE FIX: Remove all brush indicator DOM elements to prevent memory leaks
function removeAllBrushIndicators() {
    brushIndicators.forEach(indicator => {
        if (indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    });
    brushIndicators.clear();
}

// Clean up old brush indicators
function cleanupBrushIndicators(activeFingerIds) {
    brushIndicators.forEach((indicator, fingerId) => {
        if (!activeFingerIds.has(fingerId)) {
            indicator.style.display = 'none';
        }
    });
}

// Update brush size from slider
function updateBrushSize(size) {
    currentBrushSize = Math.max(MIN_BRUSH_SIZE, Math.min(MAX_BRUSH_SIZE, size));

    // Update slider handle position
    if (brushSizeSlider && brushSizeHandle) {
        const sliderHeight = brushSizeSlider.offsetHeight || 200; // Default height if not yet rendered
        const percentage = (currentBrushSize - MIN_BRUSH_SIZE) / (MAX_BRUSH_SIZE - MIN_BRUSH_SIZE);
        // Top is min (1), bottom is max (50)
        const handleY = percentage * sliderHeight;
        brushSizeHandle.style.top = handleY + 'px';
    }

    // Update value display
    if (brushSizeValue) {
        brushSizeValue.textContent = Math.round(currentBrushSize);
    }

    // Update all brush indicators size
    updateAllBrushIndicators();
}

// Handle brush size slider interaction (mouse/touch)
let isDraggingBrushSlider = false;

function handleBrushSliderMouseDown(e) {
    isDraggingBrushSlider = true;
    handleBrushSliderMove(e);
}

function handleBrushSliderMouseMove(e) {
    if (!isDraggingBrushSlider) return;
    handleBrushSliderMove(e);
}

function handleBrushSliderMouseUp() {
    isDraggingBrushSlider = false;
}

function handleBrushSliderMove(e) {
    if (!brushSizeSlider) return;

    const rect = brushSizeSlider.getBoundingClientRect();
    const y = e.clientY || (e.touches && e.touches[0].clientY);
    const sliderY = y - rect.top;
    const sliderHeight = rect.height;

    // Clamp to slider bounds
    const clampedY = Math.max(0, Math.min(sliderHeight, sliderY));

    // Convert to brush size (top is min, bottom is max)
    const percentage = clampedY / sliderHeight;
    const size = MIN_BRUSH_SIZE + (percentage * (MAX_BRUSH_SIZE - MIN_BRUSH_SIZE));

    updateBrushSize(size);
}

// Mode button event listeners
paintBtn.addEventListener('click', setPaintMode);
eraseBtn.addEventListener('click', setEraseMode);

// Brush size slider event listeners
if (brushSizeSlider && brushSizeHandle) {
    brushSizeSlider.addEventListener('mousedown', handleBrushSliderMouseDown);
    brushSizeSlider.addEventListener('touchstart', handleBrushSliderMouseDown);
    document.addEventListener('mousemove', handleBrushSliderMouseMove);
    document.addEventListener('touchmove', handleBrushSliderMouseMove);
    document.addEventListener('mouseup', handleBrushSliderMouseUp);
    document.addEventListener('touchend', handleBrushSliderMouseUp);
}

// Color picker button click
colorPickerBtn.addEventListener('click', () => {
    if (isColorPickerOpen) {
        closeColorPicker();
    } else {
        openColorPicker();
    }
});

// Initialize done button progress ring with dynamic perimeter
function initDoneButtonProgressRing() {
    const btnRect = donePickerBtn.getBoundingClientRect();
    const perimeter = 2 * (btnRect.width + btnRect.height);
    doneProgressRect.style.strokeDasharray = perimeter;
    doneProgressRect.style.strokeDashoffset = perimeter;
}

// Initialize color picker elements after DOM is ready
function initColorPickerElements() {
    initColorPicker();
    initDoneButtonProgressRing();
}

// Call after DOM is fully ready with a delay to ensure layout is complete
setTimeout(initColorPickerElements, 200);

// Re-initialize on window resize to handle fullscreen changes
window.addEventListener('resize', () => {
    initDoneButtonProgressRing();
    updateSelectorDotPosition();
    updateBrightnessSlider();
});

// Initialize MediaPipe Hands
function initMediaPipe() {
    // PERFORMANCE FIX: Close existing hands instance before creating new one
    // This prevents memory leaks from accumulated thread pools
    if (hands) {
        console.log('🧹 Closing existing MediaPipe hands instance');
        try {
            // Some versions of MediaPipe don't have close(), so wrap in try-catch
            if (typeof hands.close === 'function') {
                hands.close();
            }
        } catch (e) {
            console.warn('Could not close hands instance:', e);
        }
        hands = null;
    }

    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,  // 0 = Lite (fastest), 1 = Full
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);
}

// Initialize canvas dimensions
function initCanvasDimensions(width, height) {
    console.log('🎨 Initializing canvas dimensions:', width, 'x', height);
    console.log('Current overlay dimensions:', canvasOverlay.width, 'x', canvasOverlay.height);
    console.log('Current painting dimensions:', canvasPainting.width, 'x', canvasPainting.height);

    // Check if we actually need to resize
    const needsResize = canvasOverlay.width !== width || canvasOverlay.height !== height ||
                        canvasPainting.width !== width || canvasPainting.height !== height;

    if (!needsResize) {
        console.log('✅ Canvas dimensions already correct - skipping resize to preserve painting!');
        return;
    }

    console.log('⚠️ Canvas needs resize');

    // Save painting before resize if it exists
    let savedPainting = null;
    if (canvasPainting.width > 0 && canvasPainting.height > 0) {
        console.log('💾 Saving existing painting before resize');
        savedPainting = document.createElement('canvas');
        savedPainting.width = canvasPainting.width;
        savedPainting.height = canvasPainting.height;
        const savedCtx = savedPainting.getContext('2d');
        savedCtx.drawImage(canvasPainting, 0, 0);
    }

    // Set dimensions for both canvases (THIS CLEARS THEM AND RESETS CONTEXT STATE!)
    console.log('Resizing canvases...');
    canvasOverlay.width = width;
    canvasOverlay.height = height;
    canvasPainting.width = width;
    canvasPainting.height = height;

    // PERFORMANCE: Reset context state to defaults after resize
    // This prevents state accumulation that can cause performance degradation
    ctxOverlay.setTransform(1, 0, 0, 1, 0, 0);
    ctxPainting.setTransform(1, 0, 0, 1, 0, 0);

    // Restore painting if it existed
    if (savedPainting) {
        console.log('♻️ Restoring painting after resize');
        ctxPainting.drawImage(savedPainting, 0, 0, width, height);

        // PERFORMANCE: Explicitly clean up temporary canvas to prevent memory leak
        savedPainting.width = 0;
        savedPainting.height = 0;
        savedPainting = null;
    }

    console.log('✅ Canvas dimensions set');
}

// Check if a finger is fully extended
function isFingerExtended(landmarks, fingerTipIdx, fingerPipIdx, fingerMcpIdx) {
    const tip = landmarks[fingerTipIdx];
    const pip = landmarks[fingerPipIdx];
    const mcp = landmarks[fingerMcpIdx];

    // Finger is extended if tip is above both pip and mcp joints
    // Using a moderate threshold to reduce jitter while still requiring extension
    const tipAbovePip = tip.y < pip.y - 0.01; // 1% threshold (reduced from 2% to reduce jitter)
    const tipAboveMcp = tip.y < mcp.y;

    return tipAbovePip && tipAboveMcp;
}

// Check if thumb is extended (different logic since thumb moves horizontally)
function isThumbExtended(landmarks) {
    const thumbTip = landmarks[4];
    const thumbIp = landmarks[3];
    const thumbMcp = landmarks[2];
    const wrist = landmarks[0];

    // Calculate horizontal distance from wrist
    const tipDistanceFromWrist = Math.abs(thumbTip.x - wrist.x);
    const mcpDistanceFromWrist = Math.abs(thumbMcp.x - wrist.x);

    // Thumb is extended if tip is significantly further from wrist than mcp
    // and tip is beyond the IP joint
    const isExtended = tipDistanceFromWrist > mcpDistanceFromWrist * 1.2 &&
                      Math.abs(thumbTip.x - thumbIp.x) > 0.02;

    return isExtended;
}

// Check if hand is making a fist (all fingers closed)
function isFist(landmarks) {
    // Check if all 4 main fingers are NOT extended
    const fingers = [
        [8, 6, 5],   // index
        [12, 10, 9], // middle
        [16, 14, 13], // ring
        [20, 18, 17]  // pinky
    ];

    let closedFingers = 0;
    fingers.forEach(([tipIdx, pipIdx, mcpIdx]) => {
        const tip = landmarks[tipIdx];
        const pip = landmarks[pipIdx];
        const mcp = landmarks[mcpIdx];

        // Finger is closed if tip is NOT above pip and mcp
        const isClosed = tip.y >= pip.y - 0.01;
        if (isClosed) {
            closedFingers++;
        }
    });

    // Fist if at least 3 out of 4 fingers are closed
    return closedFingers >= 3;
}

// Check if hand is making a fist AND no fingers are extended (strict fist check)
function isStrictFist(landmarks) {
    // First check if it's a fist
    if (!isFist(landmarks)) return false;

    // Then check that no fingers are extended
    const extendedFingers = getExtendedFingers(landmarks);
    return extendedFingers.length === 0;
}

// Get fist position (use middle of palm)
function getFistPosition(landmarks) {
    // Use the middle of the palm (average of wrist and middle finger MCP)
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];

    return {
        x: (wrist.x + middleMcp.x) / 2,
        y: (wrist.y + middleMcp.y) / 2
    };
}

// Get all extended fingers and their positions
function getExtendedFingers(landmarks) {
    const extendedFingers = [];

    // REMOVED THUMB - only check the 4 main fingers (index, middle, ring, pinky)
    // Format: [tipIdx, pipIdx, mcpIdx, name]
    const fingers = [
        [8, 6, 5, 'index'],
        [12, 10, 9, 'middle'],
        [16, 14, 13, 'ring'],
        [20, 18, 17, 'pinky']
    ];

    fingers.forEach(([tipIdx, pipIdx, mcpIdx, name]) => {
        if (isFingerExtended(landmarks, tipIdx, pipIdx, mcpIdx)) {
            const tip = landmarks[tipIdx];
            extendedFingers.push({
                name: name,
                tipIdx: tipIdx,
                x: tip.x,
                y: tip.y
            });
        }
    });

    return extendedFingers;
}

// Cap alpha channel across entire canvas (called only when fingers are lifted)
// This is throttled to run at most once every 2000ms for performance
// PERFORMANCE FIX: Increased throttle time and added debouncing to prevent memory leaks
// from frequent getImageData/putImageData calls
let alphaCapTimeout = null;
function capCanvasAlphaThrottled() {
    const now = performance.now();
    if (now - lastAlphaCapTime < 2000) {
        needsAlphaCap = true; // Mark that we need to cap, but wait

        // PERFORMANCE FIX: Use debouncing instead of immediate retry
        // This prevents accumulation of pending operations
        if (!alphaCapTimeout) {
            alphaCapTimeout = setTimeout(() => {
                alphaCapTimeout = null;
                if (needsAlphaCap) {
                    capCanvasAlphaThrottled();
                }
            }, 2000 - (now - lastAlphaCapTime));
        }
        return;
    }

    lastAlphaCapTime = now;
    needsAlphaCap = false;

    // Cap alpha across entire canvas
    const imageData = ctxPainting.getImageData(0, 0, canvasPainting.width, canvasPainting.height);
    const data = imageData.data;

    // Cap alpha at 127 (50% of 255)
    const maxAlpha = 127;
    let changed = false;
    for (let i = 3; i < data.length; i += 4) {
        if (data[i] > maxAlpha) {
            data[i] = maxAlpha;
            changed = true;
        }
    }

    if (changed) {
        ctxPainting.putImageData(imageData, 0, 0);
    }
}

// Draw on painting canvas with a unique ID for each finger
function drawOnCanvas(x, y, fingerId) {
    // Don't draw if color picker is open
    if (isColorPickerOpen) {
        return;
    }

    const lastPoint = lastPaintPoints[fingerId];
    const now = Date.now();

    if (lastPoint) {
        // Calculate distance from last point
        const dx = x - lastPoint.x;
        const dy = y - lastPoint.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Only draw if the distance is reasonable (prevents jumps from detection errors)
        if (distance < MAX_LINE_DISTANCE) {
            if (isPaintMode) {
                // Paint mode - draw with current color at 50% opacity
                // PERFORMANCE FIX: Use rgba color with alpha built-in instead of globalAlpha
                // This is much faster and the browser handles blending efficiently
                const rgb = hexToRgb(currentColor);
                const colorWithAlpha = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.5)`;

                ctxPainting.globalCompositeOperation = 'source-over';
                ctxPainting.globalAlpha = 1.0;
                ctxPainting.strokeStyle = colorWithAlpha;
                ctxPainting.lineWidth = currentBrushSize * 2;
                ctxPainting.lineCap = 'round';
                ctxPainting.lineJoin = 'round';

                ctxPainting.beginPath();
                ctxPainting.moveTo(lastPoint.x, lastPoint.y);
                ctxPainting.lineTo(x, y);
                ctxPainting.stroke();

                // Mark that we need to cap alpha (will happen when fingers are lifted)
                needsAlphaCap = true;
            } else {
                // Erase mode - erase by clearing pixels
                ctxPainting.globalCompositeOperation = 'destination-out';
                ctxPainting.globalAlpha = 1.0;
                ctxPainting.lineWidth = currentBrushSize * 2;
                ctxPainting.lineCap = 'round';
                ctxPainting.lineJoin = 'round';

                ctxPainting.beginPath();
                ctxPainting.moveTo(lastPoint.x, lastPoint.y);
                ctxPainting.lineTo(x, y);
                ctxPainting.stroke();

                // Reset composite operation
                ctxPainting.globalCompositeOperation = 'source-over';
            }
        }
    }

    lastPaintPoints[fingerId] = { x, y };
    fingerLastSeen[fingerId] = now;
}

// Check if fist is hovering over fullscreen toggle button
function checkFullscreenButtonHover(landmarks) {
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return false;

    // Get button position relative to canvas
    const btnRect = fullscreenToggleBtn.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the button
    const padding = 20;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= btnRect.left - padding &&
            landmarkX <= btnRect.right + padding &&
            landmarkY >= btnRect.top - padding &&
            landmarkY <= btnRect.bottom + padding) {
            return true;
        }
    }

    return false;
}

// Check if fist is hovering over color picker button
function checkColorButtonHover(landmarks) {
    // Works in both fullscreen and normal mode
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return false;

    // Get button position relative to canvas
    const btnRect = colorPickerBtn.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the button
    const padding = 20;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= btnRect.left - padding &&
            landmarkX <= btnRect.right + padding &&
            landmarkY >= btnRect.top - padding &&
            landmarkY <= btnRect.bottom + padding) {
            return true;
        }
    }

    return false;
}



// Check if fist is hovering over done button
function checkDoneButtonHover(landmarks) {
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return false;

    // Get button position relative to canvas
    const btnRect = donePickerBtn.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the button
    const padding = 20;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= btnRect.left - padding &&
            landmarkX <= btnRect.right + padding &&
            landmarkY >= btnRect.top - padding &&
            landmarkY <= btnRect.bottom + padding) {
            return true;
        }
    }

    return false;
}

// Check if fist is hovering over paint button
function checkPaintButtonHover(landmarks) {
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return false;

    // Get button position relative to canvas
    const btnRect = paintBtn.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the button
    const padding = 20;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= btnRect.left - padding &&
            landmarkX <= btnRect.right + padding &&
            landmarkY >= btnRect.top - padding &&
            landmarkY <= btnRect.bottom + padding) {
            return true;
        }
    }

    return false;
}

// Check if fist is hovering over erase button
function checkEraseButtonHover(landmarks) {
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return false;

    // Get button position relative to canvas
    const btnRect = eraseBtn.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the button
    const padding = 20;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= btnRect.left - padding &&
            landmarkX <= btnRect.right + padding &&
            landmarkY >= btnRect.top - padding &&
            landmarkY <= btnRect.bottom + padding) {
            return true;
        }
    }

    return false;
}

// Check if fist is hovering over brush size slider and return Y position
function checkBrushSizeSliderInteraction(landmarks) {
    // Only detect if hand is making a strict fist (no fingers extended)
    if (!isStrictFist(landmarks)) return null;

    // Get slider position relative to canvas
    const sliderRect = brushSizeSlider.getBoundingClientRect();
    const canvasRect = canvasOverlay.getBoundingClientRect();

    // Check if ANY landmark of the hand is over the slider
    const padding = 30;
    for (let i = 0; i < landmarks.length; i++) {
        const landmark = landmarks[i];
        const landmarkX = canvasRect.left + (canvasRect.width - (landmark.x * canvasRect.width));
        const landmarkY = canvasRect.top + (landmark.y * canvasRect.height);

        if (landmarkX >= sliderRect.left - padding &&
            landmarkX <= sliderRect.right + padding &&
            landmarkY >= sliderRect.top - padding &&
            landmarkY <= sliderRect.bottom + padding) {
            // Return the Y position for slider interaction
            return { y: landmarkY };
        }
    }

    return null;
}

// Handle brush size slider interaction with fist
function handleBrushSizeSliderFistInteraction(y) {
    if (!brushSizeSlider) return;

    const rect = brushSizeSlider.getBoundingClientRect();
    const sliderY = y - rect.top;
    const sliderHeight = rect.height;

    // Clamp to slider bounds
    const clampedY = Math.max(0, Math.min(sliderHeight, sliderY));

    // Convert to brush size (top is min, bottom is max)
    const percentage = clampedY / sliderHeight;
    const size = MIN_BRUSH_SIZE + (percentage * (MAX_BRUSH_SIZE - MIN_BRUSH_SIZE));

    updateBrushSize(size);
}

// Check if finger is pointing at color wheel
function checkColorWheelInteraction(landmarks) {
    // Get all extended fingers
    const extendedFingers = getExtendedFingers(landmarks);

    if (extendedFingers.length === 0) return null;

    // Use the first extended finger
    const finger = extendedFingers[0];
    const canvasRect = canvasOverlay.getBoundingClientRect();
    const wheelRect = colorWheelCanvas.getBoundingClientRect();

    // Convert finger position to screen coordinates
    const fingerX = canvasRect.left + (canvasRect.width - (finger.x * canvasRect.width));
    const fingerY = canvasRect.top + (finger.y * canvasRect.height);

    // Check if finger is over the color wheel
    if (fingerX >= wheelRect.left && fingerX <= wheelRect.right &&
        fingerY >= wheelRect.top && fingerY <= wheelRect.bottom) {
        return { x: fingerX, y: fingerY };
    }

    return null;
}

// Check if finger is pointing at brightness slider
function checkBrightnessSliderInteraction(landmarks) {
    // Get all extended fingers
    const extendedFingers = getExtendedFingers(landmarks);

    if (extendedFingers.length === 0) return null;

    // Use the first extended finger
    const finger = extendedFingers[0];
    const canvasRect = canvasOverlay.getBoundingClientRect();
    const sliderRect = brightnessSlider.getBoundingClientRect();

    // Convert finger position to screen coordinates
    const fingerX = canvasRect.left + (canvasRect.width - (finger.x * canvasRect.width));
    const fingerY = canvasRect.top + (finger.y * canvasRect.height);

    // Check if finger is over the brightness slider
    if (fingerX >= sliderRect.left && fingerX <= sliderRect.right &&
        fingerY >= sliderRect.top && fingerY <= sliderRect.bottom) {
        return { x: fingerX, y: fingerY };
    }

    return null;
}

// Update color picker button progress
function updateColorButtonProgress(isHovering) {
    if (isHovering) {
        if (colorHoverStartTime === null) {
            colorHoverStartTime = performance.now();
        }

        const elapsed = performance.now() - colorHoverStartTime;
        colorHoverProgress = Math.min(elapsed / HOVER_DURATION, 1);

        // Update progress ring (170 is full circumference)
        const offset = 170 - (colorHoverProgress * 170);
        colorProgressCircle.style.strokeDashoffset = offset;

        colorPickerBtn.classList.add('hovering');

        // Open color picker when complete
        if (colorHoverProgress >= 1) {
            openColorPicker();
            colorHoverStartTime = null;
            colorHoverProgress = 0;
        }
    } else {
        colorHoverStartTime = null;
        colorHoverProgress = 0;
        colorProgressCircle.style.strokeDashoffset = 170;
        colorPickerBtn.classList.remove('hovering');
    }
}

// Update done button progress
function updateDoneButtonProgress(isHovering) {
    // Calculate perimeter dynamically based on button size
    const btnRect = donePickerBtn.getBoundingClientRect();
    const perimeter = 2 * (btnRect.width + btnRect.height);

    if (isHovering) {
        if (doneHoverStartTime === null) {
            doneHoverStartTime = performance.now();
        }

        const elapsed = performance.now() - doneHoverStartTime;
        doneHoverProgress = Math.min(elapsed / HOVER_DURATION, 1);

        // Update progress ring with dynamic perimeter
        const offset = perimeter - (doneHoverProgress * perimeter);
        doneProgressRect.style.strokeDashoffset = offset;

        donePickerBtn.classList.add('hovering');

        // Close color picker when complete
        if (doneHoverProgress >= 1) {
            closeColorPicker();
            doneHoverStartTime = null;
            doneHoverProgress = 0;
        }
    } else {
        doneHoverStartTime = null;
        doneHoverProgress = 0;
        doneProgressRect.style.strokeDashoffset = perimeter;
        donePickerBtn.classList.remove('hovering');
    }
}

// Update fullscreen button progress
function updateFullscreenButtonProgress(isHovering) {
    if (isHovering) {
        if (fullscreenHoverStartTime === null) {
            fullscreenHoverStartTime = performance.now();
        }

        const elapsed = performance.now() - fullscreenHoverStartTime;
        fullscreenHoverProgress = Math.min(elapsed / HOVER_DURATION, 1);

        // Update progress ring (170 is full circumference)
        const offset = 170 - (fullscreenHoverProgress * 170);
        fullscreenProgressCircle.style.strokeDashoffset = offset;

        fullscreenToggleBtn.classList.add('hovering');

        // Toggle fullscreen when complete
        if (fullscreenHoverProgress >= 1) {
            toggleFullscreen();
            fullscreenHoverStartTime = null;
            fullscreenHoverProgress = 0;
        }
    } else {
        fullscreenHoverStartTime = null;
        fullscreenHoverProgress = 0;
        fullscreenProgressCircle.style.strokeDashoffset = 170;
        fullscreenToggleBtn.classList.remove('hovering');
    }
}

// Update paint button progress
function updatePaintButtonProgress(isHovering) {
    if (isHovering) {
        if (paintHoverStartTime === null) {
            paintHoverStartTime = performance.now();
        }

        const elapsed = performance.now() - paintHoverStartTime;
        paintHoverProgress = Math.min(elapsed / HOVER_DURATION, 1);

        // Update progress ring (170 is full circumference)
        const offset = 170 - (paintHoverProgress * 170);
        paintProgressCircle.style.strokeDashoffset = offset;

        paintBtn.classList.add('hovering');

        // Switch to paint mode when complete
        if (paintHoverProgress >= 1) {
            setPaintMode();
            paintHoverStartTime = null;
            paintHoverProgress = 0;
        }
    } else {
        paintHoverStartTime = null;
        paintHoverProgress = 0;
        paintProgressCircle.style.strokeDashoffset = 170;
        paintBtn.classList.remove('hovering');
    }
}

// Update erase button progress
function updateEraseButtonProgress(isHovering) {
    if (isHovering) {
        if (eraseHoverStartTime === null) {
            eraseHoverStartTime = performance.now();
        }

        const elapsed = performance.now() - eraseHoverStartTime;
        eraseHoverProgress = Math.min(elapsed / HOVER_DURATION, 1);

        // Update progress ring (170 is full circumference)
        const offset = 170 - (eraseHoverProgress * 170);
        eraseProgressCircle.style.strokeDashoffset = offset;

        eraseBtn.classList.add('hovering');

        // Switch to erase mode when complete
        if (eraseHoverProgress >= 1) {
            setEraseMode();
            eraseHoverStartTime = null;
            eraseHoverProgress = 0;
        }
    } else {
        eraseHoverStartTime = null;
        eraseHoverProgress = 0;
        eraseProgressCircle.style.strokeDashoffset = 170;
        eraseBtn.classList.remove('hovering');
    }
}

// Toggle fullscreen
function toggleFullscreen() {
    const isFullscreen = videoContainer.classList.contains('fullscreen');

    if (isFullscreen) {
        // Exit fullscreen
        videoContainer.classList.remove('fullscreen');
        document.body.classList.remove('fullscreen-active');
        fullscreenIcon.textContent = '⛶'; // Fullscreen icon
    } else {
        // Enter fullscreen
        videoContainer.classList.add('fullscreen');
        document.body.classList.add('fullscreen-active');
        fullscreenIcon.textContent = '✕'; // Exit icon
    }

    fullscreenProgressCircle.style.strokeDashoffset = 170;
    fullscreenToggleBtn.classList.remove('hovering');
}

// Process results from MediaPipe
let frameCount = 0;
let lastFrameTime = performance.now();
let fps = 0;
function onResults(results) {
    if (showPaintingOnly) {
        return; // Don't process when showing painting only
    }

    // Calculate FPS
    const currentTime = performance.now();
    const delta = currentTime - lastFrameTime;
    fps = 1000 / delta;
    lastFrameTime = currentTime;

    frameCount++;
    if (frameCount % 60 === 0) { // Log every 60 frames (about once per second)
        console.log('Frame', frameCount, '- FPS:', fps.toFixed(1), '- Overlay:', canvasOverlay.width, 'x', canvasOverlay.height,
                    'Painting:', canvasPainting.width, 'x', canvasPainting.height);
    }

    // CRITICAL FIX: Clear and redraw ONLY the overlay canvas (video + hand tracking)
    // The painting canvas is NEVER cleared during tracking - this prevents flicker!
    ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    // Draw video frame (mirrored) on overlay canvas
    // FIXED: Added frame validation to prevent black screen
    if (results.image && results.image.width > 0 && results.image.height > 0) {
        // PERFORMANCE: Use save/restore instead of setTransform for better performance
        ctxOverlay.save();
        ctxOverlay.scale(-1, 1);
        ctxOverlay.drawImage(results.image, -canvasOverlay.width, 0, canvasOverlay.width, canvasOverlay.height);
        ctxOverlay.restore();
    }

    // Check for button hovers and interactions
    let isHoveringFullscreen = false;
    let isHoveringColor = false;
    let isHoveringDone = false;
    let isHoveringPaint = false;
    let isHoveringErase = false;
    let activeFingersThisFrame = new Set();

    // Draw hands and check for painting gesture
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        results.multiHandLandmarks.forEach((landmarks, handIdx) => {
            // Use current paint color for hand tracking visualization
            const color = hexToRgb(currentColor);

            // Draw hand tracking visualization on overlay canvas
            drawHand(landmarks, color);

            // If color picker is open, handle color picker interactions
            if (isColorPickerOpen) {
                // Check for color wheel interaction (finger pointing)
                const wheelPos = checkColorWheelInteraction(landmarks);
                if (wheelPos) {
                    handleColorWheelInteraction(wheelPos.x, wheelPos.y);
                }

                // Check for brightness slider interaction (finger pointing)
                const sliderPos = checkBrightnessSliderInteraction(landmarks);
                if (sliderPos) {
                    handleBrightnessSliderInteraction(sliderPos.x, sliderPos.y);
                }

                // Only check done button if NOT interacting with color wheel or brightness slider
                // This prevents accidental done button activation while pointing at controls
                if (!wheelPos && !sliderPos) {
                    // Check for done button hover (fist)
                    if (checkDoneButtonHover(landmarks)) {
                        isHoveringDone = true;
                    }
                }
            } else {
                // Normal painting mode
                // Get all extended fingers
                const extendedFingers = getExtendedFingers(landmarks);

                // Draw with each extended finger independently
                extendedFingers.forEach((finger) => {
                    // Create unique ID for this finger (hand index + finger name)
                    const fingerId = `hand${handIdx}_${finger.name}`;
                    activeFingersThisFrame.add(fingerId);

                    // Get fingertip position (mirrored)
                    const x = canvasOverlay.width - (finger.x * canvasOverlay.width);
                    const y = finger.y * canvasOverlay.height;

                    // Draw on persistent painting canvas (separate layer!)
                    drawOnCanvas(x, y, fingerId);

                    // Visual feedback - draw a circle at paint point on overlay canvas
                    // Use current color with transparency for feedback
                    const colorWithAlpha = currentColor + '80'; // Add 50% transparency
                    ctxOverlay.fillStyle = colorWithAlpha;
                    ctxOverlay.beginPath();
                    ctxOverlay.arc(x, y, 10, 0, 2 * Math.PI);
                    ctxOverlay.fill();

                    // Update brush indicator for this finger
                    try {
                        const canvasRect = canvasOverlay.getBoundingClientRect();
                        const screenX = canvasRect.left + (x / canvasOverlay.width) * canvasRect.width;
                        const screenY = canvasRect.top + (y / canvasOverlay.height) * canvasRect.height;
                        updateBrushIndicatorPosition(fingerId, screenX, screenY);
                    } catch (error) {
                        console.error('Error updating brush indicator:', error);
                    }
                });

                // Clean up indicators for fingers that are no longer active
                cleanupBrushIndicators(activeFingersThisFrame);

                // Check if any hand is hovering over buttons
                if (checkFullscreenButtonHover(landmarks)) {
                    isHoveringFullscreen = true;
                }
                if (checkColorButtonHover(landmarks)) {
                    isHoveringColor = true;
                }
                if (checkPaintButtonHover(landmarks)) {
                    isHoveringPaint = true;
                }
                if (checkEraseButtonHover(landmarks)) {
                    isHoveringErase = true;
                }

                // Check if fist is interacting with brush size slider
                const sliderInteraction = checkBrushSizeSliderInteraction(landmarks);
                if (sliderInteraction) {
                    handleBrushSizeSliderFistInteraction(sliderInteraction.y);
                }
            }
        });
    } else {
        // No hands detected, hide all brush indicators
        hideAllBrushIndicators();
    }

    // Clear last paint points for fingers that haven't been seen recently
    const now = Date.now();
    Object.keys(lastPaintPoints).forEach(fingerId => {
        if (!activeFingersThisFrame.has(fingerId)) {
            // Only clear if finger has been gone for longer than timeout
            if (fingerLastSeen[fingerId] && (now - fingerLastSeen[fingerId]) > FINGER_TIMEOUT) {
                delete lastPaintPoints[fingerId];
                delete fingerLastSeen[fingerId];
            }
        }
    });

    // Cap alpha when no fingers are actively painting (throttled for performance)
    if (activeFingersThisFrame.size === 0 && needsAlphaCap) {
        capCanvasAlphaThrottled();
    }

    // NOTE: Painting overlay is now drawn immediately after video frame (see above)
    // to prevent flashing. We don't draw it here anymore.

    // Update button progress based on mode
    if (isColorPickerOpen) {
        updateDoneButtonProgress(isHoveringDone);
    } else {
        updateFullscreenButtonProgress(isHoveringFullscreen);
        updateColorButtonProgress(isHoveringColor);
        updatePaintButtonProgress(isHoveringPaint);
        updateEraseButtonProgress(isHoveringErase);
    }
}

// Draw hand landmarks on overlay canvas
function drawHand(landmarks, color) {
    const [r, g, b] = color;

    // Hand bone connections
    const connections = [
        [0,1],[1,2],[2,3],[3,4],           // Thumb
        [0,5],[5,6],[6,7],[7,8],           // Index
        [0,9],[9,10],[10,11],[11,12],      // Middle
        [0,13],[13,14],[14,15],[15,16],    // Ring
        [0,17],[17,18],[18,19],[19,20],    // Pinky
        [5,9],[9,13],[13,17]               // Palm
    ];

    // Draw connections on overlay canvas
    ctxOverlay.strokeStyle = `rgb(${r}, ${g}, ${b})`;
    ctxOverlay.lineWidth = 3;
    ctxOverlay.beginPath();
    connections.forEach(([start, end]) => {
        const x1 = canvasOverlay.width - (landmarks[start].x * canvasOverlay.width);
        const y1 = landmarks[start].y * canvasOverlay.height;
        const x2 = canvasOverlay.width - (landmarks[end].x * canvasOverlay.width);
        const y2 = landmarks[end].y * canvasOverlay.height;
        ctxOverlay.moveTo(x1, y1);
        ctxOverlay.lineTo(x2, y2);
    });
    ctxOverlay.stroke();

    // Draw landmarks on overlay canvas
    ctxOverlay.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctxOverlay.beginPath();
    landmarks.forEach((landmark, idx) => {
        const x = canvasOverlay.width - (landmark.x * canvasOverlay.width);
        const y = landmark.y * canvasOverlay.height;
        const isFingertip = [4, 8, 12, 16, 20].includes(idx);
        const radius = isFingertip ? 6 : 4;
        ctxOverlay.moveTo(x + radius, y);
        ctxOverlay.arc(x, y, radius, 0, 2 * Math.PI);
    });
    ctxOverlay.fill();
}

// Convert hex color to RGB array
function hexToRgb(hex) {
    // Remove # if present
    hex = hex.replace('#', '');

    // Parse hex values
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);

    return [r, g, b];
}

// Stop tracking - show painting
stopBtn.addEventListener('click', () => {
    console.log('🎨 Showing painting only');
    isTracking = false;
    showPaintingOnly = true;

    if (camera) {
        camera.stop();
        camera = null;
    }

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    // Hide video and overlay canvas
    video.style.display = 'none';
    canvasOverlay.style.display = 'none';

    // Hide color picker button and paint/erase buttons when showing painting
    colorPickerBtn.style.display = 'none';
    paintBtn.style.display = 'none';
    eraseBtn.style.display = 'none';
    brushSizeSliderContainer.style.display = 'none';

    // Close color picker modal if it's open
    if (isColorPickerOpen) {
        closeColorPicker();
    }

    // Show painting canvas with white background
    canvasPainting.classList.add('painting-only');
    videoContainer.classList.add('showing-painting');

    // SAVE the painting data WITH transparency for later editing
    // PERFORMANCE: Clean up old saved data first to prevent memory leak
    if (savedPaintingData) {
        savedPaintingData.width = 0;
        savedPaintingData.height = 0;
    }

    savedPaintingData = document.createElement('canvas');
    savedPaintingData.width = canvasPainting.width;
    savedPaintingData.height = canvasPainting.height;
    const savedCtx = savedPaintingData.getContext('2d');
    savedCtx.drawImage(canvasPainting, 0, 0);

    // DRAW WHITE BACKGROUND ON THE CANVAS ITSELF
    // Fill with white
    ctxPainting.fillStyle = 'white';
    ctxPainting.fillRect(0, 0, canvasPainting.width, canvasPainting.height);

    // Draw painting back on top WITH FULL OPACITY (remove transparency)
    // Create a temporary canvas to redraw with full opacity
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasPainting.width;
    tempCanvas.height = canvasPainting.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw the saved painting data onto temp canvas
    tempCtx.drawImage(savedPaintingData, 0, 0);

    // Get image data and make all semi-transparent pixels fully opaque
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
        // If pixel has any color (not fully transparent), make it fully opaque
        // Keep RGB values the same, just set alpha to 255
        if (data[i + 3] > 0) {
            data[i + 3] = 255; // Full opacity - keep RGB values unchanged
        }
    }

    tempCtx.putImageData(imageData, 0, 0);

    // Draw the fully opaque version onto the painting canvas
    ctxPainting.drawImage(tempCanvas, 0, 0);

    // Clean up temp canvas
    tempCanvas.width = 0;
    tempCanvas.height = 0;

    // Change buttons: show "Start New Painting", "Edit Painting", and "Save as JPG"
    startBtn.textContent = 'Start New Painting';
    startBtn.disabled = false;
    startBtn.style.display = 'inline-block';
    stopBtn.disabled = true;
    editPaintingBtn.style.display = 'inline-block';
    editPaintingBtn.disabled = false;
    saveBtn.style.display = 'inline-block';
    saveBtn.disabled = false;
    statusDiv.textContent = 'Showing your painting! Start new, edit, or save.';
    statusDiv.className = 'status inactive';
});

// Start button - now handles "Start New Painting"
startBtn.addEventListener('click', async () => {
    console.log('=== START BUTTON CLICKED ===');
    const isStartingNew = startBtn.textContent === 'Start New Painting';
    console.log('Is starting new:', isStartingNew);

    // Reset button text and hide save button
    startBtn.textContent = 'Start Painting';
    editPaintingBtn.style.display = 'none';
    saveBtn.style.display = 'none';

    // Start camera first (this will initialize canvases with proper dimensions)
    await startCamera();

    // AFTER camera is started, clear painting canvas if starting new
    if (isStartingNew) {
        console.log('🗑️ CLEARING painting canvas for new painting');
        ctxPainting.clearRect(0, 0, canvasPainting.width, canvasPainting.height);
        savedPaintingData = null; // Clear saved data
    }
    console.log('=== START COMPLETE ===');
});

// Edit Painting button - brings back camera with existing painting
editPaintingBtn.addEventListener('click', async () => {
    console.log('=== EDIT BUTTON CLICKED ===');

    // OPTIMIZED: Simply restore the saved painting data (WITH transparency for editing)
    if (savedPaintingData) {
        console.log('♻️ Restoring saved painting data (with transparency for editing)');
        ctxPainting.clearRect(0, 0, canvasPainting.width, canvasPainting.height);
        ctxPainting.drawImage(savedPaintingData, 0, 0);
    }

    // Hide edit and save buttons
    editPaintingBtn.style.display = 'none';
    saveBtn.style.display = 'none';
    startBtn.textContent = 'Start Painting';

    // Start camera (painting is now transparent so you can see yourself behind it!)
    await startCamera();
    console.log('=== EDIT COMPLETE ===');
});

// Save button - downloads the painting as JPG
saveBtn.addEventListener('click', () => {
    console.log('=== SAVE BUTTON CLICKED ===');

    // Create a temporary canvas to convert to JPG with white background
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasPainting.width;
    tempCanvas.height = canvasPainting.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill with white background (JPG doesn't support transparency)
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Draw the painting on top
    tempCtx.drawImage(canvasPainting, 0, 0);

    // Convert to JPG and download
    tempCanvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        link.download = `campaint-${timestamp}.jpg`;
        link.href = url;
        link.click();

        // Clean up
        URL.revokeObjectURL(url);
        tempCanvas.width = 0;
        tempCanvas.height = 0;

        console.log('✅ CAMPAINT saved as JPG');
    }, 'image/jpeg', 0.95); // 95% quality JPG
});

// Function to start camera
async function startCamera() {
    try {
        console.log('📹 startCamera() called');

        // PERFORMANCE FIX: Clear any pending alpha cap timeout
        if (alphaCapTimeout) {
            clearTimeout(alphaCapTimeout);
            alphaCapTimeout = null;
        }

        // Clean up existing camera if it exists
        if (camera) {
            console.log('⚠️ Stopping existing camera before restart');
            camera.stop();
            camera = null;
        }

        // Clean up existing video stream if it exists
        if (video.srcObject) {
            console.log('⚠️ Stopping existing video stream before restart');
            const tracks = video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;

            // PERFORMANCE: Wait for tracks to fully stop before requesting new stream
            // This prevents resource conflicts that can cause performance degradation
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // PERFORMANCE FIX: Reset canvas context state to prevent state accumulation
        // This is critical for preventing performance degradation after multiple restarts
        ctxPainting.globalAlpha = 1.0;
        ctxPainting.globalCompositeOperation = 'source-over';
        ctxOverlay.globalAlpha = 1.0;
        ctxOverlay.globalCompositeOperation = 'source-over';

        // PERFORMANCE FIX: Clear painting state that accumulates
        lastPaintPoints = {};
        fingerLastSeen = {};
        needsAlphaCap = false;
        lastAlphaCapTime = 0;

        // PERFORMANCE FIX: Remove old brush indicator DOM elements
        removeAllBrushIndicators();

        // Initialize MediaPipe only if not already done
        // Note: Re-initializing every time causes freezing, so we reuse the instance
        if (!hands) {
            console.log('Initializing MediaPipe...');
            initMediaPipe();
        }

        console.log('Requesting camera access...');
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });

        console.log('✅ Camera stream obtained');
        console.log('Stream tracks:', stream.getVideoTracks());
        video.srcObject = stream;

        // FORCE video to play
        await video.play().catch(e => console.error('Error playing video:', e));
        console.log('✅ Video.play() called');

        video.onloadedmetadata = () => {
            console.log('=== VIDEO LOADED METADATA ===');
            console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
            console.log('Video readyState:', video.readyState);
            console.log('Video paused:', video.paused);

            // Initialize both canvas dimensions
            initCanvasDimensions(video.videoWidth, video.videoHeight);

            // Show video and both canvases
            video.style.display = 'block';
            canvasOverlay.style.display = 'block';
            canvasPainting.style.display = 'block';

            // Show color picker button and paint/erase buttons when painting is active
            colorPickerBtn.style.display = 'flex';
            paintBtn.style.display = 'flex';
            eraseBtn.style.display = 'flex';
            brushSizeSliderContainer.style.display = 'flex';

            // Remove painting-only class (this removes white background via CSS)
            canvasPainting.classList.remove('painting-only');
            videoContainer.classList.remove('showing-painting');

            showPaintingOnly = false;
            console.log('=== STARTING CAMERA ===');

            // Start MediaPipe camera
            camera = new Camera(video, {
                onFrame: async () => {
                    if (isTracking) {
                        await hands.send({ image: video });
                    }
                },
                width: video.videoWidth,
                height: video.videoHeight
            });

            camera.start();
            isTracking = true;

            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.textContent = 'Camera: Active - Point any finger up to paint!';
            statusDiv.className = 'status active';
        };

    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please ensure you have granted camera permissions.');
    }
}

// Fullscreen toggle button click (for mouse/touch interaction)
fullscreenToggleBtn.addEventListener('click', () => {
    toggleFullscreen();
});

// ESC key to exit fullscreen
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && videoContainer.classList.contains('fullscreen')) {
        exitFullscreen();
    }
});

// Click to exit fullscreen
exitFullscreenBtn.addEventListener('click', () => {
    exitFullscreen();
});

// Initialize brush size slider (with delay to ensure DOM is rendered)
if (brushSizeSlider && brushSizeHandle) {
    setTimeout(() => {
        updateBrushSize(currentBrushSize);
    }, 100);
}
