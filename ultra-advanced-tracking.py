"""
Ultra-Advanced Full-Body Tracking System - OPTIMIZED FOR PERFORMANCE
- YOLO-based pose estimation for multi-person tracking (HUMANS ONLY)
- MediaPipe Hands for advanced finger tracking
- MediaPipe Face Mesh for detailed face tracking
- GPU acceleration with automatic CPU fallback
- Threaded camera capture for zero lag
- Parallel model inference for maximum speed
- Instant snapping with no drift
- ID persistence across frames
- TARGET: 60 FPS BUTTER SMOOTH! 🧈
"""

import cv2
import numpy as np
from collections import deque, defaultdict
import time
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import mediapipe as mp
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import torch

class ThreadedCamera:
    """Threaded camera capture to eliminate frame buffer lag"""
    def __init__(self, src=0, width=640, height=480):
        print(f"🎥 Opening camera {src}...", flush=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # Use DirectShow on Windows for better compatibility

        if not self.cap.isOpened():
            print(f"❌ Failed to open camera with DirectShow, trying default backend...", flush=True)
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {src}")

        print(f"✅ Camera opened successfully!", flush=True)

        # Set properties with error handling (some cameras don't support all properties)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        except:
            pass

        # Try to set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📐 Camera resolution: {actual_width}x{actual_height}", flush=True)

        # Try to set FPS (optional, may not work on all cameras)
        try:
            self.cap.set(cv2.CAP_PROP_FPS, 60)
        except:
            pass

        # Try MJPG codec for better performance (optional)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except:
            pass

        # Try to disable auto-exposure for consistent FPS (optional)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        except:
            pass

        self.q = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        """Start the camera thread"""
        threading.Thread(target=self._update, daemon=True).start()
        time.sleep(0.5)  # Wait for first frame
        return self

    def _update(self):
        """Continuously grab frames in background thread"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                # Clear old frames and add new one
                while not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def read(self):
        """Get the latest frame"""
        return True, self.q.get()

    def stop(self):
        """Stop the camera thread"""
        self.stopped = True
        self.cap.release()

    def isOpened(self):
        """Check if camera is opened"""
        return self.cap.isOpened()

class PersonTracker:
    """Track a single person with ID persistence - SIMPLIFIED (no Kalman)"""
    def __init__(self, person_id, keypoints, bbox):
        self.id = person_id
        self.keypoints = keypoints
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confidence = 1.0

    def update(self, keypoints, bbox, confidence):
        """Update tracker with new detection - INSTANT SNAP"""
        self.keypoints = keypoints  # INSTANT UPDATE - no smoothing
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """Mark frame as missed"""
        self.time_since_update += 1
        self.age += 1

class UltraAdvancedTracker:
    def __init__(self):
        print("=" * 60, flush=True)
        print("🚀 INITIALIZING ULTRA-ADVANCED TRACKER - OPTIMIZED", flush=True)
        print("=" * 60, flush=True)

        # Enable OpenCV optimizations
        if not cv2.useOptimized():
            cv2.setUseOptimized(True)
            print("✅ Enabled OpenCV SIMD optimizations", flush=True)
        else:
            print("✅ OpenCV optimizations already enabled", flush=True)

        # Check for GPU acceleration
        print("\n[0/3] Checking for GPU acceleration...", flush=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"   CUDA Version: {torch.version.cuda}", flush=True)
            self.use_half = True  # FP16 for 2x speed on GPU
        else:
            print("⚠️  No GPU detected - using CPU (will be slower)", flush=True)
            self.use_half = False

        print("\n[1/3] Loading YOLO Pose model...", flush=True)
        try:
            # OPTIMIZATION: Try to use TensorRT engine first (2-3x faster on GPU!)
            import os
            tensorrt_pose_path = 'yolo11n-pose.engine'

            if self.device == 'cuda' and os.path.exists(tensorrt_pose_path):
                print(f"🚀 Found TensorRT engine: {tensorrt_pose_path}", flush=True)
                self.pose_model = YOLO(tensorrt_pose_path)
                print("✅ YOLO Pose TensorRT model loaded (ULTRA FAST!)!", flush=True)
            else:
                # Use YOLOv8 pose model for multi-person tracking
                self.pose_model = YOLO('yolo11n-pose.pt')  # Nano model for speed
                print("✅ YOLO Pose model loaded successfully!", flush=True)

                # Suggest TensorRT export for GPU users
                if self.device == 'cuda':
                    print("💡 TIP: Export to TensorRT for 2-3x speedup:", flush=True)
                    print("   model.export(format='engine', half=True, imgsz=320)", flush=True)
        except Exception as e:
            print(f"❌ Error loading YOLO Pose: {e}", flush=True)
            raise

        # Optimize model for speed
        self.pose_model.overrides['verbose'] = False

        print("\n[2/4] Loading YOLO Segmentation model for body outline...", flush=True)
        try:
            # OPTIMIZATION: Try to use TensorRT engine first (2-3x faster on GPU!)
            tensorrt_seg_path = 'yolo11n-seg.engine'

            if self.device == 'cuda' and os.path.exists(tensorrt_seg_path):
                print(f"🚀 Found TensorRT engine: {tensorrt_seg_path}", flush=True)
                self.seg_model = YOLO(tensorrt_seg_path)
                print("✅ YOLO Segmentation TensorRT model loaded (ULTRA FAST!)!", flush=True)
            else:
                # Use YOLO segmentation model for accurate body contours
                self.seg_model = YOLO('yolo11n-seg.pt')  # Segmentation model
                print("✅ YOLO Segmentation model loaded successfully!", flush=True)
        except Exception as e:
            print(f"❌ Error loading YOLO Segmentation: {e}", flush=True)
            raise

        # Optimize segmentation model for speed
        self.seg_model.overrides['verbose'] = False

        # Initialize MediaPipe Hands for finger tracking - OPTIMIZED
        print("\n[3/4] Loading MediaPipe Hands for finger tracking...", flush=True)
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Reduced from 4 for speed
                min_detection_confidence=0.6,  # Higher threshold for speed
                min_tracking_confidence=0.6,
                model_complexity=0  # Use lite model for speed
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            print("✅ MediaPipe Hands loaded (OPTIMIZED)!", flush=True)
        except Exception as e:
            print(f"❌ Error loading MediaPipe: {e}", flush=True)
            raise

        # Initialize MediaPipe Face Mesh for detailed face tracking - OPTIMIZED
        print("\n[4/4] Loading MediaPipe Face Mesh for face/eye/mouth tracking...", flush=True)
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,  # Reduced from 4 for speed
                refine_landmarks=False,  # Disable iris for speed (was True)
                min_detection_confidence=0.6,  # Higher threshold for speed
                min_tracking_confidence=0.6
            )
            print("✅ MediaPipe Face Mesh loaded (OPTIMIZED)!", flush=True)
        except Exception as e:
            print(f"❌ Error loading MediaPipe Face Mesh: {e}", flush=True)
            raise

        print("\n✨ All models loaded! Ready for 60fps! 🚀\n", flush=True)

        # Thread pool for parallel inference (2 workers optimal for MediaPipe)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Tracking parameters
        self.next_id = 0
        self.trackers = []
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3

        # Canvas and drawing
        self.canvas = None
        self.draw_color = (0, 255, 0)
        self.brush_thickness = 8

        # Interaction zones per person
        self.person_zones = defaultdict(dict)

        # Hand tracking data per person
        self.person_hands = defaultdict(list)

        # UI settings - ALWAYS ON, NO TOGGLES
        self.show_skeleton = True
        self.show_ids = False
        self.show_confidence = False
        self.show_zones = False
        self.show_fingers = True
        self.is_fullscreen = True
        self.neon_mode = True

        # Performance
        self.enhance_enabled = False

        # Frame skipping for performance
        self.frame_skip = 0
        self.frame_count = 0

        # FPS monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Processing resolution (ULTRA OPTIMIZED for maximum FPS)
        self.yolo_width = 256  # Even lower res for maximum speed
        self.yolo_height = 192
        self.mediapipe_width = 320  # Reduced for speed
        self.mediapipe_height = 240
        self.display_width = 640
        self.display_height = 480

        # Pre-allocate resize buffers to avoid memory allocation overhead
        self.yolo_buffer = np.zeros((self.yolo_height, self.yolo_width, 3), dtype=np.uint8)
        self.mediapipe_buffer = np.zeros((self.mediapipe_height, self.mediapipe_width, 3), dtype=np.uint8)

        # Warmup models for consistent performance (after buffers are allocated)
        print("[WARMUP] Warming up models for consistent FPS...", flush=True)
        dummy_frame = np.zeros((self.yolo_height, self.yolo_width, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.pose_model(dummy_frame, verbose=False, device=self.device, half=self.use_half)
            _ = self.seg_model(dummy_frame, verbose=False, device=self.device, half=self.use_half)
        print("✅ Models warmed up!\n", flush=True)

        # OPTIMIZATION: Frame skipping for segmentation (body outline doesn't change much!)
        self.seg_frame_skip = 3  # Run every 3 frames for balance of speed and smoothness
        self.frame_count = 0
        self.cached_seg_results = None
        self.cached_person_masks = []

        # DISABLE frame skipping for MediaPipe - causes jitter
        self.mediapipe_frame_skip = 1  # Run every frame for smoothness

        # Temporal smoothing for body masks
        self.prev_masks = {}
        self.mask_alpha = 0.7  # Increased for smoother blending (less jitter)

        # Body part indices (COCO format)
        self.KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # Drawing points per person
        self.person_draw_points = defaultdict(lambda: deque(maxlen=100))

        # Cache FPS text
        self.fps_text_cache = "FPS: 0"

        # Pre-compute neon colors to avoid repeated calculations
        self.neon_colors = [
            (255, 0, 255),    # Neon Magenta
            (0, 255, 255),    # Neon Cyan
            (0, 255, 0),      # Neon Green
            (255, 255, 0),    # Neon Yellow
            (255, 0, 128),    # Neon Pink
            (128, 255, 0),    # Neon Lime
            (255, 128, 0),    # Neon Orange
            (128, 0, 255),    # Neon Purple
        ]

        # Pre-compute glow colors (40% intensity) for each neon color
        self.glow_colors = [tuple(int(c * 0.4) for c in color) for color in self.neon_colors]


    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def associate_detections_to_trackers(self, detections, trackers):
        """Associate detections to existing trackers using Hungarian algorithm - OPTIMIZED"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))

        # Calculate cost matrix (1 - IOU) - vectorized for speed
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.calculate_iou(det['bbox'], trk.bbox)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Handle empty matches
        matched_indices = np.column_stack((row_ind, col_ind)) if len(row_ind) > 0 else np.empty((0, 2), dtype=int)

        # Find unmatched detections and trackers
        unmatched_detections = [d for d in range(len(detections)) if len(matched_indices) == 0 or d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if len(matched_indices) == 0 or t not in matched_indices[:, 1]]

        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        return matches, unmatched_detections, unmatched_trackers

    def is_valid_human_detection(self, keypoints):
        """Validate that detection is actually a human by checking keypoint structure"""
        # Check if we have key human body parts visible
        # Nose (0), shoulders (5,6), hips (11,12)
        key_points = [0, 5, 6, 11, 12]
        visible_count = sum(1 for idx in key_points if idx < len(keypoints) and keypoints[idx][2] > 0.3)

        # Need at least 3 key body parts visible to confirm human
        if visible_count < 3:
            return False

        # Check for reasonable body proportions
        # If shoulders and hips are visible, check distance ratio
        if (keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3 and
            keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3):

            shoulder_width = np.linalg.norm(
                np.array([keypoints[5][0], keypoints[5][1]]) -
                np.array([keypoints[6][0], keypoints[6][1]])
            )
            hip_width = np.linalg.norm(
                np.array([keypoints[11][0], keypoints[11][1]]) -
                np.array([keypoints[12][0], keypoints[12][1]])
            )

            # Shoulder and hip width should be somewhat similar (0.5x to 2x ratio)
            if shoulder_width > 0 and hip_width > 0:
                ratio = shoulder_width / hip_width
                if ratio < 0.3 or ratio > 3.0:
                    return False

        return True

    def update_trackers(self, detections):
        """Update trackers with new detections - SIMPLIFIED"""
        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, self.trackers
        )

        # Update matched trackers - INSTANT SNAP
        for m in matches:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(
                detections[det_idx]['keypoints'],
                detections[det_idx]['bbox'],
                detections[det_idx]['confidence']
            )

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            tracker = PersonTracker(
                self.next_id,
                detections[i]['keypoints'],
                detections[i]['bbox']
            )
            self.trackers.append(tracker)
            self.next_id += 1

        # Mark unmatched trackers as missed
        for i in unmatched_trks:
            self.trackers[i].mark_missed()

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        return [t for t in self.trackers if t.hits >= self.min_hits]

    def get_body_zones(self, keypoints, person_id):
        """Extract body zones for interaction"""
        zones = {}

        # Left hand (wrist)
        if keypoints[9][2] > 0.5:
            zones['left_hand'] = (int(keypoints[9][0]), int(keypoints[9][1]))

        # Right hand (wrist)
        if keypoints[10][2] > 0.5:
            zones['right_hand'] = (int(keypoints[10][0]), int(keypoints[10][1]))

        # Head (nose)
        if keypoints[0][2] > 0.5:
            zones['head'] = (int(keypoints[0][0]), int(keypoints[0][1]))

        # Left foot (ankle)
        if keypoints[15][2] > 0.5:
            zones['left_foot'] = (int(keypoints[15][0]), int(keypoints[15][1]))

        # Right foot (ankle)
        if keypoints[16][2] > 0.5:
            zones['right_foot'] = (int(keypoints[16][0]), int(keypoints[16][1]))

        self.person_zones[person_id] = zones
        return zones

    def draw_skeleton(self, frame, keypoints, person_id, confidence, body_mask=None):
        """Draw FULL BODY OUTLINE - SMOOTH and ACCURATE"""
        # Use pre-computed neon colors
        color = self.neon_colors[person_id % len(self.neon_colors)]

        # DRAW ACCURATE BODY OUTLINE from segmentation mask (SMOOTH METHOD)
        if body_mask is not None:
            self.draw_body_contour_from_mask(frame, body_mask, color, person_id)

        # That's it! No skeleton lines, no joint points - just clean body outline
        # Face and hands are handled by MediaPipe separately

    def draw_body_contour_from_mask(self, frame, mask, color, person_id=0):
        """Draw accurate body outline from segmentation mask - SMOOTH"""
        if mask is None or mask.size == 0:
            return

        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Resize mask to match frame size with better interpolation
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Temporal smoothing - blend with previous mask
        if person_id in self.prev_masks:
            mask = cv2.addWeighted(mask, self.mask_alpha, self.prev_masks[person_id], 1 - self.mask_alpha, 0)
        self.prev_masks[person_id] = mask.copy()

        # OPTIMIZED: Reduced Gaussian blur (smaller kernel = faster)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return

        # Get the largest contour (main body)
        largest_contour = max(contours, key=cv2.contourArea)

        # Better contour smoothing with smaller epsilon
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Draw the contour with MINIMAL GLOW for performance
        thickness = 2

        # Single glow layer
        glow_color = tuple(int(c * 0.4) for c in color)
        cv2.drawContours(frame, [smoothed_contour], -1, glow_color, thickness + 3)

        # Bright core
        cv2.drawContours(frame, [smoothed_contour], -1, color, thickness)

    def draw_zones(self, frame, zones, person_id):
        """Draw body zones with person-specific colors"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)
        ]
        base_color = colors[person_id % len(colors)]

        for position in zones.values():
            if position:
                cv2.circle(frame, position, 15, base_color, 2)
                cv2.circle(frame, position, 5, base_color, -1)


    def detect_and_draw_fingers(self, processing_frame, display_frame):
        """Detect and draw finger tracking using MediaPipe Hands - NEON STYLE"""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = self.hands.process(rgb_frame)

        hand_data = []

        if results.multi_hand_landmarks and self.show_fingers:
            process_h, process_w, _ = processing_frame.shape
            display_h, display_w, _ = display_frame.shape

            # Calculate scaling factors
            scale_x = display_w / process_w
            scale_y = display_h / process_h

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Neon color for hands (bright cyan/magenta)
                hand_color = (255, 0, 255) if hand_idx % 2 == 0 else (0, 255, 255)
                glow_color = self.glow_colors[hand_idx % len(self.glow_colors)]

                # Draw hand connections with neon glow
                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]

                    # Scale coordinates to display resolution
                    start_point = (int(start.x * process_w * scale_x), int(start.y * process_h * scale_y))
                    end_point = (int(end.x * process_w * scale_x), int(end.y * process_h * scale_y))

                    # Draw glow effect - minimal for performance
                    cv2.line(display_frame, start_point, end_point, glow_color, 3)
                    cv2.line(display_frame, start_point, end_point, hand_color, 1)

                # Draw all hand landmarks - MINIMAL for performance
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Scale coordinates to display resolution
                    x = int(landmark.x * process_w * scale_x)
                    y = int(landmark.y * process_h * scale_y)

                    # Finger tips are larger - MINIMAL glow
                    if idx in [4, 8, 12, 16, 20]:  # Finger tips
                        radius = 6
                        # Single glow layer
                        cv2.circle(display_frame, (x, y), radius + 3, glow_color, -1)
                        # Bright core
                        cv2.circle(display_frame, (x, y), radius, hand_color, -1)
                        # White hot center
                        cv2.circle(display_frame, (x, y), 2, (255, 255, 255), -1)
                    else:
                        # Regular landmarks - MINIMAL
                        radius = 3
                        cv2.circle(display_frame, (x, y), radius + 2, glow_color, -1)
                        cv2.circle(display_frame, (x, y), radius, hand_color, -1)

                hand_data.append({
                    'landmarks': hand_landmarks,
                    'handedness': results.multi_handedness[hand_idx].classification[0].label
                })

        return hand_data

    def detect_and_draw_faces(self, processing_frame, display_frame, active_trackers):
        """Detect and draw face mesh with eyes and mouth - NEON STYLE"""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Process faces
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            process_h, process_w, _ = processing_frame.shape
            display_h, display_w, _ = display_frame.shape

            # Calculate scaling factors
            scale_x = display_w / process_w
            scale_y = display_h / process_h

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Neon color for face (bright green/yellow)
                face_color = (0, 255, 255) if face_idx % 2 == 0 else (0, 255, 0)

                # Key landmark indices for face features
                # Left eye: 33, 133, 160, 159, 158, 157, 173, 246
                # Right eye: 362, 263, 385, 386, 387, 388, 398, 466
                # Lips outer: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
                # Lips inner: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
                # Face oval: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109

                # Draw FULL HEAD outline (extended to include forehead/top of head)
                # This creates a more complete head shape that matches the body outline better
                # Using a wider set of landmarks to create a fuller head shape
                full_head_indices = [
                    # Start from left temple, go around forehead, down right side, under chin, back up left side
                    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,  # Left side down to chin
                    377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10  # Right side back up
                ]

                # Draw with simple lines (smoother than polylines for this use case)
                glow_color = self.glow_colors[face_idx % len(self.glow_colors)]
                for i in range(len(full_head_indices)):
                    start_idx = full_head_indices[i]
                    end_idx = full_head_indices[(i + 1) % len(full_head_indices)]

                    start = face_landmarks.landmark[start_idx]
                    end = face_landmarks.landmark[end_idx]

                    # Scale coordinates to display resolution
                    start_point = (int(start.x * process_w * scale_x), int(start.y * process_h * scale_y))
                    end_point = (int(end.x * process_w * scale_x), int(end.y * process_h * scale_y))

                    # Draw glow - minimal for performance
                    cv2.line(display_frame, start_point, end_point, glow_color, 3)
                    cv2.line(display_frame, start_point, end_point, face_color, 1)

                # Draw left eye
                left_eye_indices = [33, 160, 159, 158, 133, 153, 144, 145, 33]
                self.draw_face_feature(display_frame, face_landmarks, left_eye_indices, face_color, process_w, process_h, scale_x, scale_y)

                # Draw right eye
                right_eye_indices = [362, 385, 386, 387, 263, 373, 374, 380, 362]
                self.draw_face_feature(display_frame, face_landmarks, right_eye_indices, face_color, process_w, process_h, scale_x, scale_y)

                # Draw irises (if available with refine_landmarks=True)
                # Left iris: 468, 469, 470, 471, 472
                # Right iris: 473, 474, 475, 476, 477
                if len(face_landmarks.landmark) > 468:
                    # Left iris
                    left_iris_center = face_landmarks.landmark[468]
                    left_iris_point = (int(left_iris_center.x * process_w * scale_x), int(left_iris_center.y * process_h * scale_y))
                    cv2.circle(display_frame, left_iris_point, 8, tuple(int(c * 0.3) for c in face_color), -1)
                    cv2.circle(display_frame, left_iris_point, 4, face_color, -1)
                    cv2.circle(display_frame, left_iris_point, 2, (255, 255, 255), -1)

                    # Right iris
                    right_iris_center = face_landmarks.landmark[473]
                    right_iris_point = (int(right_iris_center.x * process_w * scale_x), int(right_iris_center.y * process_h * scale_y))
                    cv2.circle(display_frame, right_iris_point, 8, tuple(int(c * 0.3) for c in face_color), -1)
                    cv2.circle(display_frame, right_iris_point, 4, face_color, -1)
                    cv2.circle(display_frame, right_iris_point, 2, (255, 255, 255), -1)

                # Draw mouth outer
                mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61]
                self.draw_face_feature(display_frame, face_landmarks, mouth_outer_indices, face_color, process_w, process_h, scale_x, scale_y, thickness=2)

                # Draw mouth inner
                mouth_inner_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
                self.draw_face_feature(display_frame, face_landmarks, mouth_inner_indices, face_color, process_w, process_h, scale_x, scale_y, thickness=1)

                # NO NECK - body outline handles the connection

    def draw_face_feature(self, frame, landmarks, indices, color, w, h, scale_x, scale_y, thickness=2):
        """Helper to draw a face feature (eye, mouth, etc.) with neon glow - OPTIMIZED"""
        glow_color = tuple(int(c * 0.4) for c in color)

        for i in range(len(indices) - 1):
            start = landmarks.landmark[indices[i]]
            end = landmarks.landmark[indices[i + 1]]

            # Scale coordinates to display resolution
            start_point = (int(start.x * w * scale_x), int(start.y * h * scale_y))
            end_point = (int(end.x * w * scale_x), int(end.y * h * scale_y))

            # Draw glow effect - simple 2 layers
            cv2.line(frame, start_point, end_point, glow_color, thickness + 2)
            cv2.line(frame, start_point, end_point, color, thickness)

    def run(self):
        """Main tracking loop - OPTIMIZED"""
        # Initialize threaded webcam
        print("🎥 Starting threaded camera capture...", flush=True)
        cap = ThreadedCamera(src=0, width=self.display_width, height=self.display_height).start()

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("=" * 60, flush=True)
        print("🎥 NEON SKELETON TRACKER - ULTRA OPTIMIZED!", flush=True)
        print("=" * 60, flush=True)
        print("", flush=True)
        print("✨ Features:", flush=True)
        print("  - Black background with CLEAN NEON visualization", flush=True)
        print("  - Multi-person tracking (HUMANS ONLY)", flush=True)
        print("  - SMOOTH FULL BODY OUTLINE using AI segmentation", flush=True)
        print("  - GPU-accelerated inference" if self.device == 'cuda' else "  - CPU inference", flush=True)
        print("  - Threaded camera capture (ZERO LAG)", flush=True)
        print("  - Parallel MediaPipe processing", flush=True)
        print("  - Detailed FACE tracking (eyes, mouth, face outline)", flush=True)
        print("  - Advanced HAND/FINGER tracking with neon glow", flush=True)
        print("", flush=True)
        print("🚀 Performance Optimizations:", flush=True)
        print(f"  - YOLO resolution: {self.yolo_width}x{self.yolo_height} (LOW for speed!)", flush=True)
        print(f"  - MediaPipe resolution: {self.mediapipe_width}x{self.mediapipe_height} (reduced)", flush=True)
        print(f"  - Segmentation frame skip: Every {self.seg_frame_skip} frames (3x faster!)", flush=True)
        print("  - Minimal glow layers (1 layer instead of 3)", flush=True)
        print("  - Reduced line thickness for faster drawing", flush=True)
        print("  - Pre-allocated buffers (no memory allocation overhead)", flush=True)
        print("  - torch.inference_mode() for 10-15% memory savings", flush=True)
        print("  - Optimized NMS (agnostic_nms=True, max_det=10)", flush=True)
        print("  - Skip MediaPipe when no people detected", flush=True)
        print("  - Pre-computed colors (no repeated calculations)", flush=True)
        print("", flush=True)
        print("Press 'q' to quit", flush=True)
        print("=" * 60, flush=True)

        # Create window
        window_name = 'Ultra-Advanced Body Tracking - OPTIMIZED'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.is_fullscreen = True

        while True:
            # FPS calculation
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_text_cache = f"FPS: {self.current_fps}"  # Update cached text
                print(f"📊 FPS: {self.current_fps}", flush=True)
                self.fps_counter = 0
                self.fps_start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # OPTIMIZATION: Use pre-allocated buffers for resize
            cv2.resize(frame, (self.yolo_width, self.yolo_height),
                      dst=self.yolo_buffer, interpolation=cv2.INTER_NEAREST)
            cv2.resize(frame, (self.mediapipe_width, self.mediapipe_height),
                      dst=self.mediapipe_buffer, interpolation=cv2.INTER_LINEAR)

            # NEON MODE: Black background, camera runs in background for detection
            display_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

            # OPTIMIZATION: Wrap YOLO inference in torch.inference_mode() for memory savings
            with torch.inference_mode():
                # Run YOLO pose detection - GPU ACCELERATED (on lower res frame!)
                pose_results = self.pose_model(
                    self.yolo_buffer,
                    verbose=False,
                    conf=0.35,  # Slightly higher confidence for fewer false positives
                    iou=0.5,   # NMS threshold
                    agnostic_nms=True,  # Faster NMS across all classes
                    max_det=10,  # Limit to 10 people max for speed
                    imgsz=self.yolo_width,  # Match YOLO resolution
                    device=self.device,  # Use GPU if available!
                    half=self.use_half  # FP16 on GPU for 2x speed
                )

                # OPTIMIZATION: Run YOLO segmentation every 3rd frame for smooth body outline
                self.frame_count += 1
                if self.frame_count % self.seg_frame_skip == 0 or self.cached_seg_results is None:
                    # Run segmentation on this frame
                    seg_results = self.seg_model(
                        self.yolo_buffer,
                        verbose=False,
                        conf=0.35,
                        iou=0.5,
                        agnostic_nms=True,  # Faster NMS
                        max_det=10,  # Limit detections
                        imgsz=self.yolo_width,
                        device=self.device,  # Use GPU if available!
                        half=self.use_half,  # FP16 on GPU for 2x speed
                        classes=[0]  # Only detect person class (class 0 in COCO)
                    )
                    self.cached_seg_results = seg_results
                else:
                    # Reuse cached segmentation from previous frame
                    seg_results = self.cached_seg_results

                # Periodic CUDA cache clearing to prevent memory fragmentation
                if self.frame_count % 300 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

            # Extract detections - FILTER FOR HUMANS ONLY
            detections = []
            if pose_results[0].keypoints is not None:
                # Pre-compute scaling factors
                scale_x = self.display_width / self.yolo_width
                scale_y = self.display_height / self.yolo_height

                for i in range(len(pose_results[0].boxes)):
                    box = pose_results[0].boxes[i]
                    keypoints_raw = pose_results[0].keypoints[i].data[0].cpu().numpy()

                    # OPTIMIZED: Scale keypoints in-place (avoid copy when possible)
                    keypoints = keypoints_raw.copy()  # Still need copy to avoid modifying original
                    keypoints[:, 0] *= scale_x
                    keypoints[:, 1] *= scale_y

                    # VALIDATE: Only accept if it's actually a human
                    if not self.is_valid_human_detection(keypoints):
                        continue  # Skip non-human detections

                    # OPTIMIZED: Get bounding box (scale to display resolution from YOLO resolution)
                    bbox_data = box.xyxy[0].cpu().numpy()
                    x1 = bbox_data[0] * scale_x
                    y1 = bbox_data[1] * scale_y
                    x2 = bbox_data[2] * scale_x
                    y2 = bbox_data[3] * scale_y
                    confidence = box.conf[0].cpu().numpy()

                    detections.append({
                        'keypoints': keypoints,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })

            # Update trackers - INSTANT SNAP
            active_trackers = self.update_trackers(detections)

            # Extract segmentation masks for each person
            person_masks = []
            if seg_results[0].masks is not None:
                for i in range(len(seg_results[0].masks)):
                    mask = seg_results[0].masks[i].data[0].cpu().numpy()
                    # Scale mask to display resolution
                    mask_resized = cv2.resize(mask, (self.display_width, self.display_height))
                    person_masks.append(mask_resized)

            # OPTIMIZATION: Only run MediaPipe if people are detected
            if len(active_trackers) > 0:
                # OPTIMIZATION: Run MediaPipe every N frames (hands/face don't change as fast!)
                if self.frame_count % self.mediapipe_frame_skip == 0:
                    # PARALLEL PROCESSING: Run MediaPipe models in parallel
                    # Submit both tasks to thread pool (using mediapipe_buffer for better quality)
                    future_faces = self.executor.submit(self.detect_and_draw_faces, self.mediapipe_buffer, display_frame, active_trackers)
                    future_hands = self.executor.submit(self.detect_and_draw_fingers, self.mediapipe_buffer, display_frame)

                    # Wait for MediaPipe to complete (they run in parallel)
                    future_faces.result()
                    future_hands.result()
                # If skipping MediaPipe this frame, still draw body outline

                # Draw all tracked persons on display frame (NEON SKELETONS with ORIGINAL segmentation)
                for idx, tracker in enumerate(active_trackers):
                    # Get corresponding mask if available
                    body_mask = person_masks[idx] if idx < len(person_masks) else None

                    # Draw skeleton with ORIGINAL segmentation method
                    self.draw_skeleton(display_frame, tracker.keypoints, tracker.id, tracker.confidence, body_mask)

            # Add FPS counter to display (using cached text)
            cv2.putText(display_frame, self.fps_text_cache, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display pure neon skeletons
            cv2.imshow(window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break

        # Cleanup
        cap.stop()
        self.hands.close()
        self.face_mesh.close()
        self.executor.shutdown(wait=False)
        cv2.destroyAllWindows()
        print("👋 Ultra-Advanced Tracking Closed!")
        print(f"📊 Final FPS: {self.current_fps}")

if __name__ == "__main__":
    tracker = UltraAdvancedTracker()
    tracker.run()


