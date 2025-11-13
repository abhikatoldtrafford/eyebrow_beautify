# Eyebrow Stencil API

**REST API for AI-powered eyebrow stencil generation using YOLO + MediaPipe**

*Version: 6.0 | Production Ready*

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /mnt/g/eyebrow
pip install ultralytics opencv-python numpy scipy mediapipe fastapi uvicorn pillow
```

### 2. Verify Model Weights

```bash
# Check that YOLO model exists (59MB via Git LFS)
ls eyebrow_training/eyebrow_recommended/weights/best.pt

# If missing, pull from Git LFS
git lfs pull
```

### 3. Start the API Server

```bash
# From the eyebrow directory
./start_api.sh

# OR manually:
python3 -m uvicorn api.api_main:app --host 0.0.0.0 --port 8000 --reload
```

**Wait for this output:**
```
Loading YOLO model...
✓ YOLO model loaded successfully
✓ MediaPipe available
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Access Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## 📡 API Endpoints (19 Total)

### Health & Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint (welcome message) |
| `/health` | GET | Check service status, model availability |
| `/config` | GET | Get current configuration |
| `/config` | POST | Update configuration |

---

### Detection Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect/yolo` | POST | YOLO detection only (file upload) |
| `/detect/yolo/base64` | POST | YOLO detection only (Base64 JSON) |
| `/detect/mediapipe` | POST | MediaPipe landmarks only (file upload) |
| `/detect/mediapipe/base64` | POST | MediaPipe landmarks only (Base64 JSON) |

**Use Case**: Testing individual detection sources, debugging, component integration

---

### Preprocessing Endpoint

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/preprocess` | POST | Face validation and rotation detection (Base64 JSON) |

**Returns**:
- Face validity (eyes, eyebrows, quality checks)
- Rotation angle estimation (multi-source)
- Asymmetry detection (angle, position, span)
- Detailed rejection reasons if invalid

---

### Stencil Extraction Endpoints ⭐ MAIN

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/beautify/base64` | POST | **Main endpoint** - Complete 4-phase polygon extraction (Base64 JSON) |
| `/beautify/submit-edit` | POST | Submit user-edited polygon for validation |

**What it does**:
1. **Phase 0**: Face preprocessing & validation
2. **Phase 1**: Extract YOLO polygon from mask
3. **Phase 2**: Check YOLO-MediaPipe alignment (IoU threshold: 0.3)
4. **Phase 3**: Merge or fallback to MediaPipe-only
5. **Phase 4**: Validate and return precise polygon boundaries

**Returns**: Simplified polygon points ready for interactive editing

---

### Stencil Library Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stencils/save` | POST | Save stencil to library with metadata |
| `/stencils/list` | GET | List all saved stencils (optional filtering) |
| `/stencils/{id}` | GET | Get specific stencil by ID |
| `/stencils/{id}` | DELETE | Delete stencil from library |

**Storage**: File-based JSON in `stencil_data/` directory

---

### Adjustment Endpoints (Legacy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/adjust/thickness/increase` | POST | Increase eyebrow thickness by 5% (morphological) |
| `/adjust/thickness/decrease` | POST | Decrease eyebrow thickness by 5% |
| `/adjust/span/increase` | POST | Increase eyebrow span by 5% (tail only) |
| `/adjust/span/decrease` | POST | Decrease eyebrow span by 5% |

**Note**: These work on binary masks. The frontend now uses interactive polygon editing instead.

---

### Generation Endpoint (Placeholder)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate/sd-beautify` | POST | Stable Diffusion enhancement (not yet implemented) |

---

## 📝 Usage Examples

### Example 1: Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "mediapipe_available": true,
  "version": "6.0"
}
```

---

### Example 2: Extract Stencil Polygons (Main Workflow)

**Python Client:**

```python
import requests
import base64
from PIL import Image
import io

API_URL = "http://localhost:8000"

def extract_stencil(image_path):
    """Extract eyebrow stencil polygons from image."""

    # Read and encode image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Call main stencil extraction endpoint
    response = requests.post(
        f"{API_URL}/beautify/base64",
        json={"image_base64": img_base64}
    )

    result = response.json()

    if result['success']:
        print(f"✓ Extracted {len(result['stencils'])} stencils")

        for stencil in result['stencils']:
            side = stencil['metadata']['side']
            points = stencil['polygon']['points']
            print(f"  {side.upper()}: {len(points)} control points")

            # Validation metrics
            validation = stencil['validation']
            print(f"    IoU: {validation['iou']:.2f}")
            print(f"    Method: {validation['method']}")
            print(f"    Valid: {validation['is_valid']}")
    else:
        print(f"✗ Error: {result['message']}")

    return result

# Usage
result = extract_stencil("photo.jpg")
```

**Response Structure:**
```json
{
  "success": true,
  "message": "Successfully extracted 2 stencil(s)",
  "stencils": [
    {
      "polygon": {
        "points": [[127, 730], [145, 725], ...],
        "point_count": 15,
        "method": "merged",
        "bbox": {
          "x1": 127.0, "y1": 725.0,
          "x2": 353.0, "y2": 793.0
        }
      },
      "metadata": {
        "side": "left",
        "yolo_confidence": 0.95,
        "mediapipe_detected": true,
        "image_shape": [800, 600]
      },
      "validation": {
        "iou": 0.85,
        "alignment_score": 0.92,
        "method": "merged",
        "is_valid": true,
        "yolo_point_count": 18,
        "mp_point_count": 10
      }
    },
    {
      "polygon": {...},
      "metadata": {"side": "right", ...},
      "validation": {...}
    }
  ],
  "preprocessing": {
    "valid": true,
    "rotation_angle": 2.3,
    "rotation_corrected": true
  },
  "processing_time_ms": 453.12,
  "image_shape": [800, 600]
}
```

---

### Example 3: Face Preprocessing Check

```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoAAAANS..."
  }'
```

**Response:**
```json
{
  "success": true,
  "valid": true,
  "rotation_angle": 3.5,
  "rotation_sources": {
    "mediapipe_eyes": 3.2,
    "yolo_eyes": 3.8,
    "yolo_eye_box": 3.4
  },
  "face_validation": {
    "eyes_valid": true,
    "eyebrows_valid": true,
    "quality_valid": true,
    "rotation_valid": true
  },
  "asymmetry_detection": {
    "angle_asymmetry": 2.1,
    "position_asymmetry": 0.03,
    "span_asymmetry": 0.08
  },
  "rejection_reason": null,
  "processing_time_ms": 234.56
}
```

---

### Example 4: Save Stencil to Library

```python
def save_stencil_to_library(polygon_points, side, image_base64):
    """Save extracted polygon to stencil library."""

    response = requests.post(
        f"{API_URL}/stencils/save",
        json={
            "polygon": polygon_points,
            "side": side,
            "image_base64": image_base64,
            "metadata": {
                "tags": ["custom", "client_name"],
                "notes": "Perfect arch shape"
            }
        }
    )

    result = response.json()

    if result['success']:
        print(f"✓ Saved stencil: {result['stencil_id']}")

    return result

# Usage
save_stencil_to_library(
    polygon_points=[[127, 730], [145, 725], ...],
    side="left",
    image_base64="iVBORw0KGgoAAAANS..."
)
```

**Response:**
```json
{
  "success": true,
  "message": "Stencil saved successfully",
  "stencil_id": "a3b7c9d1-e4f5-6789-0abc-def123456789",
  "filename": "stencil_a3b7c9d1-e4f5-6789-0abc-def123456789.json"
}
```

---

### Example 5: List Saved Stencils

```bash
# Get all stencils
curl http://localhost:8000/stencils/list

# Filter by side
curl "http://localhost:8000/stencils/list?side=left"
```

**Response:**
```json
{
  "success": true,
  "count": 12,
  "stencils": [
    {
      "id": "a3b7c9d1-e4f5-6789-0abc-def123456789",
      "side": "left",
      "created_at": "2025-01-13T10:30:00Z",
      "tags": ["custom", "client_name"],
      "point_count": 15,
      "preview_url": "/stencils/a3b7c9d1-e4f5-6789-0abc-def123456789"
    },
    ...
  ]
}
```

---

### Example 6: YOLO Detection Only

```bash
curl -X POST "http://localhost:8000/detect/yolo?conf_threshold=0.25" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Detection successful",
  "detections": {
    "eyebrows": [
      {
        "class_id": 2,
        "class_name": "eyebrows",
        "confidence": 0.95,
        "box": {"x1": 127.0, "y1": 730.0, "x2": 353.0, "y2": 793.0},
        "mask_area": 5234,
        "mask_centroid": [223, 760],
        "mask_base64": "iVBORw0KGgoAAAANS..."
      }
    ],
    "eye": [...],
    "eye_box": [...],
    "hair": [...]
  },
  "processing_time_ms": 156.78,
  "image_shape": [800, 600]
}
```

---

### Example 7: MediaPipe Detection Only

```python
def get_mediapipe_landmarks(image_path):
    """Get MediaPipe facial landmarks."""

    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    response = requests.post(
        f"{API_URL}/detect/mediapipe/base64",
        json={"image_base64": img_base64}
    )

    result = response.json()

    if result['success']:
        left_eyebrow = result['landmarks']['left_eyebrow']
        print(f"Left eyebrow: {len(left_eyebrow['points'])} landmarks")

    return result
```

**Response:**
```json
{
  "success": true,
  "message": "Detection successful",
  "landmarks": {
    "left_eyebrow": {
      "points": [[127, 730], [145, 725], ..., [353, 790]],
      "indices": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
      "center": [223, 760],
      "bbox": {"x1": 127.0, "y1": 730.0, "x2": 353.0, "y2": 793.0}
    },
    "right_eyebrow": {...},
    "left_eye": {...},
    "right_eye": {...}
  },
  "processing_time_ms": 89.34,
  "image_shape": [800, 600]
}
```

---

## 🔧 Configuration Parameters

### Get Current Configuration

```bash
curl http://localhost:8000/config
```

### Update Configuration

```bash
curl -X POST "http://localhost:8000/config" \
  -H "Content-Type: application/json" \
  -d '{
    "yolo_conf_threshold": 0.30,
    "alignment_iou_threshold": 0.35,
    "yolo_simplify_epsilon": 0.004
  }'
```

### All Available Parameters

```json
{
  "yolo_conf_threshold": 0.25,
  "yolo_simplify_epsilon": 0.005,
  "alignment_iou_threshold": 0.3,
  "alignment_distance_threshold": 20.0,
  "min_polygon_points": 5,
  "max_polygon_points": 50,
  "grounding_method": "insert_mp",
  "fallback_to_mp": true,
  "enable_preprocessing": true,
  "reject_invalid_faces": true,
  "auto_correct_rotation": true,
  "min_rotation_threshold": 1.0,
  "max_rotation_angle": 30.0
}
```

**Parameter Descriptions:**

- **`yolo_conf_threshold`**: YOLO detection confidence (0.1-0.9). Lower = more detections, higher = only confident detections.
- **`yolo_simplify_epsilon`**: Douglas-Peucker simplification (0.001-0.01). Lower = more points, higher = smoother polygon.
- **`alignment_iou_threshold`**: YOLO-MediaPipe IoU threshold (0.2-0.5). Determines if sources agree.
- **`alignment_distance_threshold`**: Max pixel distance for alignment (10-50 pixels).
- **`grounding_method`**: How to merge YOLO + MediaPipe (`insert_mp`, `weighted_avg`, `boundary_trace`).
- **`fallback_to_mp`**: Use MediaPipe-only if alignment fails (true/false).
- **`enable_preprocessing`**: Enable face validation and rotation correction (true/false).
- **`min_rotation_threshold`**: Minimum angle to correct (0.5-5.0 degrees).
- **`max_rotation_angle`**: Maximum acceptable rotation before rejection (10-45 degrees).

---

## 🎨 JavaScript/TypeScript Client

```typescript
interface StencilExtractionResult {
  success: boolean;
  message: string;
  stencils: Array<{
    polygon: {
      points: [number, number][];
      point_count: number;
      method: string;
    };
    metadata: {
      side: string;
      yolo_confidence: number;
      mediapipe_detected: boolean;
    };
    validation: {
      iou: number;
      is_valid: boolean;
      method: string;
    };
  }>;
  processing_time_ms: number;
}

async function extractStencil(imageFile: File): Promise<StencilExtractionResult> {
  // Convert file to base64
  const base64 = await fileToBase64(imageFile);

  const response = await fetch('http://localhost:8000/beautify/base64', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: base64 })
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return await response.json();
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = (reader.result as string).split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Usage with React
const handleImageUpload = async (file: File) => {
  try {
    const result = await extractStencil(file);

    if (result.success) {
      console.log(`Extracted ${result.stencils.length} stencils`);
      result.stencils.forEach(stencil => {
        console.log(`${stencil.metadata.side}: ${stencil.polygon.point_count} points`);
      });
    }
  } catch (error) {
    console.error('Extraction failed:', error);
  }
};
```

---

## 🐛 Troubleshooting

### Model Not Loading

**Error:** `"YOLO model not loaded. Service unavailable."`

**Solution:**
```bash
# Check model path
ls eyebrow_training/eyebrow_recommended/weights/best.pt

# If missing, pull from Git LFS
git lfs install
git lfs pull

# Verify model size (should be ~59MB)
du -h eyebrow_training/eyebrow_recommended/weights/best.pt
```

---

### MediaPipe Not Available

**Error:** `"MediaPipe not available"`

**Solution:**
```bash
pip install mediapipe
```

---

### Port Already in Use

**Error:** `"Address already in use (8000)"`

**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Restart API
./start_api.sh
```

---

### Face Validation Fails

**Error:** `"Face validation failed: rotation angle exceeds threshold"`

**Solution:**
- Use front-facing photo (not profile view)
- Ensure face is straight (rotation <30°)
- Check that both eyebrows are visible
- Verify good lighting and image quality

**Or disable preprocessing:**
```python
response = requests.post(
    f"{API_URL}/beautify/base64",
    json={
        "image_base64": img_base64,
        "config": {"enable_preprocessing": False}
    }
)
```

---

## 📊 Performance Characteristics

### Processing Time (CPU, 800×600 image)

| Operation | Time | Notes |
|-----------|------|-------|
| YOLO detection | 100-150ms | Depends on image size |
| MediaPipe detection | 50-100ms | Face mesh extraction |
| Polygon extraction | 30-50ms | Douglas-Peucker simplification |
| Face preprocessing | 200-350ms | Multi-source rotation detection |
| **Total (with preprocessing)** | **400-700ms** | End-to-end stencil extraction |
| **Total (without preprocessing)** | **200-350ms** | Faster for validated images |

### Memory Usage

| Component | Memory |
|-----------|--------|
| YOLO model | ~20MB |
| MediaPipe model | ~10MB |
| FastAPI app | ~50MB |
| Peak usage | ~100MB |

### Accuracy

| Metric | Typical Range |
|--------|---------------|
| YOLO-MediaPipe IoU | 0.6-0.9 (when aligned) |
| Alignment success rate | 75-85% |
| MediaPipe fallback rate | 15-25% |
| Final polygon points | 10-30 points |

---

## 🔐 Security Considerations (Production)

1. **CORS**: Configure allowed origins in `api_main.py`
2. **Rate Limiting**: Add rate limiting middleware
3. **File Size Limits**: Enforce maximum upload size (default: 10MB)
4. **Authentication**: Implement API key or JWT authentication
5. **Input Validation**: All inputs validated via Pydantic models
6. **HTTPS**: Use reverse proxy (Nginx) with SSL/TLS

---

## 🐳 Docker Deployment

**Create `Dockerfile`:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download model weights via Git LFS
RUN git lfs pull

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t eyebrow-stencil-api .
docker run -p 8000:8000 eyebrow-stencil-api
```

---

## 📚 Additional Resources

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Main README**: `../README.md` (User guide)
- **Technical Reference**: `../CLAUDE.md` (Algorithm deep dive)
- **Frontend**: `../frontend/` (React application)

---

## 🎯 API Workflow Summary

```
1. Upload Image → POST /beautify/base64
   ↓
2. Backend Processing:
   - Phase 0: Face preprocessing (rotation correction)
   - Phase 1: YOLO detection → binary mask → polygon
   - Phase 2: MediaPipe landmarks → 10 points
   - Phase 3: Check alignment (IoU threshold)
   - Phase 4: Merge or fallback → simplified polygon
   ↓
3. Return Polygon Points (10-30 vertices)
   ↓
4. Frontend: Interactive Canvas Editing (zoom, pan, drag points)
   ↓
5. Save to Library → POST /stencils/save
   ↓
6. Browse Library → GET /stencils/list
```

---

**API Version: 6.0 | Ready for Production** 🚀

Start the server and visit http://localhost:8000/docs to explore the interactive API documentation!
