# Eyebrow Beautification API

**REST API for eyebrow detection and beautification using YOLO + MediaPipe**

Version: 1.0.0

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /mnt/g/eyebrow
pip install -r api/requirements.txt
```

### 2. Start the Server

```bash
# From the eyebrow directory
python -m uvicorn api.api_main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## 📡 API Endpoints

### Health & Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check service status and model availability |
| `/config` | GET | Get current configuration |
| `/config` | POST | Update configuration |

### Detection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect/yolo` | POST | YOLO detection only (file upload) |
| `/detect/yolo/base64` | POST | YOLO detection only (base64 JSON) |
| `/detect/mediapipe` | POST | MediaPipe landmarks only (file upload) |
| `/detect/mediapipe/base64` | POST | MediaPipe landmarks only (base64 JSON) |

### Beautification

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/beautify` | POST | Complete pipeline with preprocessing (file upload) |
| `/beautify/base64` | POST | Complete pipeline with preprocessing (base64 JSON) |
| `/beautify/submit-edit` | POST | Submit user-edited eyebrow mask |

### Adjustments

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/adjust/thickness/increase` | POST | Increase eyebrow thickness by 5% |
| `/adjust/thickness/decrease` | POST | Decrease eyebrow thickness by 5% |
| `/adjust/span/increase` | POST | Increase eyebrow span by 5% (tail only) |
| `/adjust/span/decrease` | POST | Decrease eyebrow span by 5% (tail only) |

### Generation (Placeholder)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate/sd-beautify` | POST | Stable Diffusion enhancement (not yet implemented) |

> **Note on Preprocessing:** The `/beautify` endpoints automatically include comprehensive face preprocessing:
> - **Face rotation detection** (4 sources: MediaPipe eyes, YOLO eyes, YOLO eye_box, YOLO eyebrows)
> - **Automatic rotation correction** (threshold: 2.5°, with source agreement validation)
> - **Face quality validation** (eyes, eyebrows, MediaPipe landmark coverage)
> - **Asymmetry detection** (warns if left/right eyebrows are significantly different)

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
  "version": "1.0.0"
}
```

---

### Example 2: Beautify Eyebrows (File Upload)

```bash
curl -X POST "http://localhost:8000/beautify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 2 eyebrow(s)",
  "eyebrows": [
    {
      "side": "left",
      "validation": {
        "mp_coverage": 100.0,
        "mp_coverage_pass": true,
        "eye_distance_pct": 5.2,
        "eye_distance_pass": true,
        "aspect_ratio": 7.1,
        "aspect_ratio_pass": true,
        "eye_overlap": 0,
        "eye_overlap_pass": true,
        "expansion_ratio": 1.30,
        "expansion_ratio_pass": true,
        "overall_pass": true
      },
      "metadata": {
        "yolo_confidence": 0.95,
        "yolo_area": 5234,
        "final_area": 6804,
        "has_eye": true,
        "has_eye_box": true,
        "hair_regions": 0,
        "has_mediapipe": true
      },
      "original_mask_base64": "iVBORw0KGgoAAAANS...",
      "final_mask_base64": "iVBORw0KGgoAAAANS..."
    },
    {
      "side": "right",
      ...
    }
  ],
  "processing_time_ms": 1523.45,
  "image_shape": [2048, 944]
}
```

---

### Example 3: Beautify with Custom Configuration

```bash
curl -X POST "http://localhost:8000/beautify?config=%7B%22min_mp_coverage%22%3A90%7D" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

Or update global config first:

```bash
curl -X POST "http://localhost:8000/config" \
  -H "Content-Type: application/json" \
  -d '{
    "yolo_conf_threshold": 0.3,
    "min_mp_coverage": 90.0,
    "straightening_threshold": 3.0
  }'
```

---

### Example 4: YOLO Detection Only

```bash
curl -X POST "http://localhost:8000/detect/yolo?conf_threshold=0.25" \
  -H "accept: application/json" \
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
        "box_width": 226.0,
        "box_height": 63.0,
        "center": [240, 761],
        "mask_area": 5234,
        "mask_centroid": [223, 760],
        "mask_base64": "iVBORw0KGgoAAAANS..."
      }
    ],
    "eye": [...],
    "eye_box": [...],
    "hair": [...]
  },
  "processing_time_ms": 456.78,
  "image_shape": [2048, 944]
}
```

---

### Example 5: MediaPipe Detection Only

```bash
curl -X POST "http://localhost:8000/detect/mediapipe?conf_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
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
  "processing_time_ms": 234.56,
  "image_shape": [2048, 944]
}
```

---

## 🐍 Python Client Example

```python
import requests
import base64
from PIL import Image
import io

# API endpoint
API_URL = "http://localhost:8000"

def beautify_image_from_file(image_path):
    """Upload image file and get beautified eyebrows."""

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{API_URL}/beautify",
            files=files
        )

    return response.json()

def beautify_image_from_base64(image_path):
    """Send image as base64 in JSON."""

    # Read and encode image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Send request
    response = requests.post(
        f"{API_URL}/beautify/base64",
        json={
            "image_base64": img_base64,
            "return_masks": True
        }
    )

    return response.json()

def decode_mask_from_response(mask_base64):
    """Decode base64 mask to numpy array."""
    import numpy as np
    import cv2

    mask_bytes = base64.b64decode(mask_base64)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    mask_array = np.array(mask_img)

    return mask_array

# Usage
if __name__ == "__main__":
    # Beautify image
    result = beautify_image_from_file("test_image.jpg")

    if result['success']:
        print(f"Processed {len(result['eyebrows'])} eyebrows")

        for eyebrow in result['eyebrows']:
            print(f"\n{eyebrow['side'].upper()} Eyebrow:")
            print(f"  MP Coverage: {eyebrow['validation']['mp_coverage']:.1f}%")
            print(f"  Overall Pass: {eyebrow['validation']['overall_pass']}")

            # Decode and save mask
            final_mask = decode_mask_from_response(eyebrow['final_mask_base64'])
            Image.fromarray(final_mask).save(f"mask_{eyebrow['side']}.png")
    else:
        print(f"Error: {result['message']}")
```

---

## 🔧 Configuration Parameters

All parameters can be configured via `/config` endpoint:

```json
{
  "yolo_conf_threshold": 0.25,
  "mediapipe_conf_threshold": 0.5,
  "straightening_threshold": 5.0,
  "min_mp_coverage": 80.0,
  "eye_dist_range": [4.0, 8.0],
  "aspect_ratio_range": [3.0, 10.0],
  "expansion_range": [0.9, 2.0],
  "min_arch_thickness_pct": 0.015,
  "connection_thickness_pct": 0.01,
  "eye_buffer_kernel": [15, 15],
  "eye_buffer_iterations": 2,
  "hair_overlap_threshold": 0.15,
  "hair_distance_threshold": 0.3,
  "close_kernel": [7, 7],
  "open_kernel": [5, 5],
  "gaussian_kernel": [9, 9],
  "gaussian_sigma": 2.0
}
```

**Parameter Descriptions:**

- `yolo_conf_threshold`: YOLO detection confidence (0.1-0.9)
- `mediapipe_conf_threshold`: MediaPipe detection confidence (0.1-0.9)
- `straightening_threshold`: Auto-straighten face if angle > this (degrees)
- `min_mp_coverage`: Minimum MediaPipe landmark coverage required (%)
- `eye_dist_range`: Valid range for eyebrow-eye distance (% of image height)
- `aspect_ratio_range`: Valid range for eyebrow aspect ratio (width/height)
- `expansion_range`: Valid range for expansion ratio (final_area / original_area)

---

## 🎨 JavaScript/TypeScript Example

```typescript
async function beautifyEyebrows(imageFile: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch('http://localhost:8000/beautify', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return await response.json();
}

// Usage with file input
const fileInput = document.querySelector('#imageInput') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    try {
      const result = await beautifyEyebrows(file);
      console.log('Eyebrows processed:', result);

      // Display masks
      result.eyebrows.forEach((eyebrow: any) => {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${eyebrow.final_mask_base64}`;
        document.body.appendChild(img);
      });
    } catch (error) {
      console.error('Error:', error);
    }
  }
});
```

---

## 🐳 Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t eyebrow-api .
docker run -p 8000:8000 eyebrow-api
```

---

## 🔐 Security Considerations (Production)

1. **CORS:** Configure allowed origins properly
2. **Rate Limiting:** Add rate limiting middleware
3. **Authentication:** Add API key or JWT authentication
4. **File Size Limits:** Enforce maximum upload size
5. **Input Validation:** Validate all inputs
6. **HTTPS:** Use HTTPS in production

---

## 🐛 Troubleshooting

### Model Not Loading

**Error:** "YOLO model not loaded. Service unavailable."

**Solution:**
- Check model path: `eyebrow_training/eyebrow_recommended/weights/best.pt`
- Ensure model file exists
- Check server logs for loading errors

### MediaPipe Not Available

**Error:** "MediaPipe not available"

**Solution:**
```bash
pip install mediapipe
```

### Out of Memory

**Solution:**
- Reduce image size before uploading
- Use GPU inference for YOLO
- Increase server memory

---

## 📊 Performance Tips

1. **Model Caching:** Model is loaded once on startup (singleton pattern)
2. **Async Processing:** FastAPI handles requests asynchronously
3. **GPU Acceleration:** YOLO can use GPU automatically if CUDA available
4. **Image Optimization:** Resize large images before uploading

---

## 📚 Additional Resources

- **API Documentation:** `/docs` (Swagger UI)
- **Algorithm Specification:** `../COMPREHENSIVE_IMPLEMENTATION_PLAN.md`
- **Core API Documentation:** `../API_DOCUMENTATION.md`
- **Critical Assessment:** `../CRITICAL_ASSESSMENT.md`

---

**API Ready!** 🚀

Start the server and visit http://localhost:8000/docs to explore the interactive API documentation.
