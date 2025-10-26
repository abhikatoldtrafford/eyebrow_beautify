"""
Streamlit App Configuration
Constants, API URLs, and default settings
"""

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds

# UI Colors (RGB)
COLORS = {
    'left_eyebrow': (255, 0, 0),      # Red
    'right_eyebrow': (0, 0, 255),     # Blue
    'yolo_detection': (0, 255, 0),    # Green
    'mediapipe_points': (255, 255, 0), # Yellow
    'eye': (255, 0, 255),             # Magenta
    'hair': (0, 255, 255),            # Cyan
}

# Default opacity for overlays
DEFAULT_OPACITY = 0.5

# MediaPipe keypoint size
MP_POINT_RADIUS = 3

# Adjustment increments
THICKNESS_INCREMENT = 0.05  # 5% per click
SPAN_INCREMENT = 0.05       # 5% per click

# Manual edit limits
ROTATION_RANGE = (-45, 45)    # degrees
SCALE_RANGE = (0.5, 1.5)      # 50% to 150%

# API default thresholds (should match beautify.py DEFAULT_CONFIG)
API_DETECTION_DEFAULTS = {
    'yolo_conf_threshold': 0.25,
    'mediapipe_conf_threshold': 0.3,  # Lowered from 0.5 for better detection
}

# SD Default parameters
SD_DEFAULTS = {
    'prompt': 'natural, well-groomed eyebrows, high detail, photorealistic, symmetric',
    'negative_prompt': 'blurry, distorted, unnatural, asymmetric, sparse, patchy',
    'strength': 0.7,
    'guidance_scale': 7.5,
    'num_inference_steps': 50,
}

# Image size limits
MAX_IMAGE_SIZE = (1920, 1080)  # Max width, height
MIN_IMAGE_SIZE = (200, 200)     # Min width, height

# Session state keys
SESSION_KEYS = {
    'initialized': False,
    'original_image': None,
    'original_image_b64': None,
    'yolo_detections': None,
    'mediapipe_landmarks': None,
    'eyebrows': None,
    'current_masks': {'left': None, 'right': None},
    'edit_history': [],
    'clicks': {'left': {'thickness': 0, 'span': 0}, 'right': {'thickness': 0, 'span': 0}},
    'finalized': False,
    'sd_result': None,
    'api_healthy': False,
}

# UI Messages
MESSAGES = {
    'upload_prompt': "Upload an image to begin eyebrow beautification",
    'processing': "Processing your image...",
    'api_error': "Failed to connect to API. Make sure the server is running.",
    'no_eyebrows': "No eyebrows detected. Please try another image.",
    'success': "Eyebrows detected successfully!",
    'finalized': "Masks finalized! Ready for Phase 2 enhancement.",
    'sd_processing': "Enhancing with Stable Diffusion... This may take a while.",
}

# Feature flags
FEATURES = {
    'manual_edit': True,      # Enable manual editing mode
    'auto_edit': True,        # Enable auto adjustment mode
    'sd_enhancement': True,   # Enable SD Phase 2 (currently placeholder)
    'download': True,         # Enable download results
    'advanced_settings': True, # Show advanced API settings
}
