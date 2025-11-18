"""
Web-Based Body Tracking System
- Flask web server for browser-based camera access
- Real-time body tracking with YOLO and MediaPipe
- WebSocket streaming for low latency
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

class WebBodyTracker:
    def __init__(self):
        print("🚀 Initializing Hand Tracker...")

        # Load MediaPipe Hands optimized for speed
        print("Loading MediaPipe Hands...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower for faster detection
            min_tracking_confidence=0.5,   # Lower for faster tracking
            model_complexity=0  # Fastest model (0 = lite, 1 = full)
        )

        # Camera
        self.camera = None
        self.camera_lock = Lock()

        # Processing settings
        self.process_width = 640
        self.process_height = 480

        print("✅ Hand Tracker initialized!")
    
    def start_camera(self):
        """Start the webcam"""
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.process_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.process_height)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                return True
            return False
    
    def stop_camera(self):
        """Stop the webcam"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
    
    def process_frame(self, frame):
        """Process a single frame and return hand tracking data"""
        # Don't flip here - the frontend handles mirroring for display

        tracking_data = {
            'hands': []
        }

        # Process hands (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append({
                        'x': float(landmark.x * frame.shape[1]),
                        'y': float(landmark.y * frame.shape[0]),
                        'z': float(landmark.z)
                    })

                # Get handedness (left or right)
                handedness = "Right"
                if hand_results.multi_handedness:
                    handedness = hand_results.multi_handedness[idx].classification[0].label

                tracking_data['hands'].append({
                    'landmarks': landmarks_list,
                    'handedness': handedness
                })

        return tracking_data

# Global tracker instance
tracker = WebBodyTracker()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera"""
    tracker.start_camera()
    return jsonify({'status': 'success'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera"""
    tracker.stop_camera()
    return jsonify({'status': 'success'})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame from the browser and return tracking data"""
    try:
        # Get image from request
        file = request.files.get('frame')
        if not file:
            return jsonify({'error': 'No frame provided'}), 400

        # Get image dimensions from request (for accurate coordinate mapping)
        width = request.form.get('width', type=int)
        height = request.form.get('height', type=int)

        # Convert to OpenCV format
        image = Image.open(BytesIO(file.read()))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Use actual frame dimensions if not provided
        if width is None:
            width = frame.shape[1]
        if height is None:
            height = frame.shape[0]

        # Process frame
        tracking_data = tracker.process_frame(frame)

        # Add image dimensions to response for accurate coordinate mapping
        tracking_data['image_width'] = width
        tracking_data['image_height'] = height

        return jsonify(tracking_data)

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Hand Tracker Server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("👋 Show your hands to the camera for tracking!")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

