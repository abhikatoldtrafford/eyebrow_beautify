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

# Brush/Eraser Configuration
BRUSH_CONFIG = {
    'default_brush_size': 10,
    'min_brush_size': 1,
    'max_brush_size': 50,
    'default_opacity': 0.6,
    'brush_color': "#FF0000",  # Red
    'eraser_color': "#FFFFFF",  # White
}

# Zoom Configuration
ZOOM_CONFIG = {
    'min_zoom': 0.5,
    'max_zoom': 4.0,
    'zoom_step': 0.1,
    'default_zoom': 1.0,
}

# Undo/Redo Configuration
UNDO_CONFIG = {
    'max_history': 50,
    'enable_redo': True,
}

# Mobile Configuration
MOBILE_CONFIG = {
    'phone_breakpoint': 768,     # px - viewport width for phone/tablet
    'tablet_breakpoint': 1024,   # px - viewport width for tablet/desktop
    'touch_debounce': 500,       # ms - debounce for touch events
    'min_touch_target': 44,      # px - minimum touch target size
}

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
    # Existing keys
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

    # Brush/eraser mode
    'brush_tool': '🖌️ Brush',
    'brush_size': BRUSH_CONFIG['default_brush_size'],
    'brush_opacity': BRUSH_CONFIG['default_opacity'],
    'canvas_data': None,

    # Auto adjust mode
    'last_thickness': {'left': 0, 'right': 0},
    'last_span': {'left': 0, 'right': 0},
    'thickness_left': 0,
    'thickness_right': 0,
    'span_left': 0,
    'span_right': 0,
    'adjustment_cache': {},  # LRU cache for API results

    # Zoom/pan
    'zoom_level': ZOOM_CONFIG['default_zoom'],
    'pan_offset': (0, 0),

    # Undo/redo
    'undo_stack': [],
    'redo_stack': [],
    'max_undo': UNDO_CONFIG['max_history'],

    # UI state
    'edit_tab': 'Auto Adjust',
    'preview_mode': 'Overlay',
    'preview_opacity': DEFAULT_OPACITY,

    # Mobile
    'is_mobile': False,
    'touch_start': None,
    'last_api_call': 0,  # Timestamp for debouncing
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
