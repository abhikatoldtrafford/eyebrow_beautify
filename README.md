# Brow Stencil App

**Create custom eyebrow stencils from photos with AI-powered detection and interactive editing**

*Version 6.0 | Production Ready*

---

## 🎯 What is This?

The **Brow Stencil App** is an end-to-end system that:
1. **Detects eyebrows** from photos using AI (YOLO + MediaPipe)
2. **Extracts precise polygon boundaries** for stencil creation
3. **Lets you edit** polygons interactively on a canvas (zoom, pan, drag points)
4. **Saves to a library** for later use or export

Perfect for:
- Makeup artists creating custom eyebrow stencils
- Beauty salons offering personalized eyebrow shaping
- Product designers prototyping eyebrow templates
- Researchers studying facial features and symmetry

---

## ✨ Key Features

✅ **AI-Powered Detection** - Combines YOLO segmentation with MediaPipe landmarks for accurate boundaries
✅ **Interactive Canvas Editor** - Zoom, pan, drag control points to perfect your stencil
✅ **Face Validation** - Automatically detects and rejects invalid faces (rotation, poor quality)
✅ **Stencil Library** - Save, browse, filter, and delete your custom stencils
✅ **REST API** - 19 endpoints for integration with other apps
✅ **Modern UI** - React frontend with smooth interactions
✅ **Production Ready** - Tested and documented for deployment

---

## 🖼️ Screenshots

### Upload Page
Upload a photo and the system automatically detects eyebrows, validates the face, and extracts polygon boundaries.

### Editor Page (Interactive Canvas)
- **Green polygon** = Left eyebrow
- **Red polygon** = Right eyebrow
- **White circles** = Control points (drag to move)
- **Mouse wheel** = Zoom in/out
- **Drag canvas** = Pan around
- **Double-click edge** = Add control point
- **Click point + Delete key** = Remove point

### Library Page
Browse all saved stencils, filter by side (left/right), preview thumbnails, and delete unwanted stencils.

---

## 🚀 Quick Start

### Prerequisites

Install these before starting:

```bash
# Python 3.8+ (check version)
python3 --version

# Node.js 16+ and npm 8+ (check versions)
node --version
npm --version

# Git LFS (for downloading model weights)
git lfs version
```

**Don't have these installed?**

<details>
<summary>Click here for installation instructions</summary>

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip nodejs npm git-lfs
```

**macOS (using Homebrew):**
```bash
brew install python node git-lfs
```

**Windows:**
- Use WSL2 (Windows Subsystem for Linux) and follow Ubuntu instructions
- Or download installers from official websites

</details>

---

### Installation

**Step 1: Clone Repository**

```bash
git clone https://github.com/your-repo/eyebrow.git
cd eyebrow

# Download model weights (59MB via Git LFS)
git lfs pull

# Verify model exists
ls eyebrow_training/eyebrow_recommended/weights/best.pt
# Should show: best.pt
```

**Step 2: Install Python Dependencies**

```bash
pip install ultralytics opencv-python numpy scipy mediapipe fastapi uvicorn pillow

# Verify installation
python -c "import cv2, mediapipe, ultralytics; print('✓ All imports successful')"
```

**Step 3: Install Frontend Dependencies**

```bash
cd frontend
npm install
cd ..
```

**Step 4: Start the Backend API** (Terminal 1)

```bash
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

**Step 5: Start the Frontend** (Terminal 2)

```bash
cd frontend
npm start
```

**Wait for this output:**
```
Compiled successfully!
You can now view frontend in the browser.
  Local:            http://localhost:3000
```

**Step 6: Open in Browser**

Navigate to: **http://localhost:3000**

---

## 📖 How to Use

### 1. Upload a Photo

1. Click **"Upload"** in the navigation
2. Drag and drop a photo OR click to browse files
3. **Photo requirements**:
   - Front-facing face
   - Both eyebrows visible
   - Good lighting
   - Not rotated more than 30°
   - JPG or PNG format (max 10MB)

**What happens next:**
- System validates face quality (rotation, eyes, eyebrows)
- If invalid, shows rejection reason
- If valid, extracts left and right eyebrow polygons
- Automatically redirects to Editor page

---

### 2. Edit Your Stencils

**You are now in the Interactive Canvas Editor!**

**Controls:**
- **Zoom**: Mouse wheel scroll (zoom range: 0.5x - 5.0x)
- **Pan**: Drag the canvas (not the control points)
- **Move control point**: Drag white circles to adjust polygon shape
- **Add control point**: Double-click on polygon edge
- **Delete control point**: Click white circle + press Delete key
- **Reset view**: Click "Fit to Screen" button

**Polygon colors:**
- Green = Left eyebrow
- Red = Right eyebrow

**Tips:**
- Zoom in (mouse wheel up) for precise editing
- Pan around to focus on specific areas
- Add points for more detail on curves
- Delete points to simplify straight sections

---

### 3. Save to Library

1. After editing, click **"Save Stencil"** button
2. System saves both left and right eyebrows
3. Confirmation message appears
4. Optionally navigate to Library to verify

---

### 4. Browse Your Library

1. Click **"Library"** in navigation
2. See all saved stencils as cards
3. **Filter by side**: Click "Left" or "Right" tabs
4. **Delete stencil**: Click "Delete" button on card
5. **View details**: Click on card to see full polygon data

---

## 🎛️ System Architecture

### Technology Stack

**Backend (Python)**
- FastAPI - REST API framework
- YOLO (Ultralytics) - AI eyebrow detection
- MediaPipe (Google) - Facial landmark detection
- OpenCV - Image processing
- NumPy/SciPy - Polygon operations

**Frontend (React)**
- React 19 - UI framework
- Konva.js - Canvas rendering and interaction
- React Router - Navigation
- Axios - HTTP client
- React Dropzone - File upload

**Data Storage**
- File-based JSON - Stencil library storage
- Git LFS - Model weights storage (59MB)

### Endpoints

**19 API Endpoints** (full list at http://localhost:8000/docs):

| Category | Endpoints |
|----------|-----------|
| Health & Config | `GET /health`, `GET /config`, `POST /config` |
| Detection | `POST /detect/yolo`, `/detect/mediapipe` (+ Base64 variants) |
| Preprocessing | `POST /preprocess` |
| Stencil Extraction | `POST /beautify/base64` (⭐ main endpoint) |
| Stencil Library | `POST /stencils/save`, `GET /stencils/list`, `GET /stencils/{id}`, `DELETE /stencils/{id}` |
| Adjustments | `POST /adjust/thickness/increase|decrease`, `POST /adjust/span/increase|decrease` |
| Other | `POST /beautify/submit-edit`, `POST /generate/sd-beautify` |

**View interactive docs:** http://localhost:8000/docs (Swagger UI)

---

## 🗂️ Project Structure

```
eyebrow/
├── backend/                           # Python backend
│   ├── api/
│   │   ├── api_main.py               # FastAPI app (19 endpoints)
│   │   ├── api_models.py             # Request/response schemas
│   │   └── api_utils.py              # Utilities (Base64, etc.)
│   ├── stencil_extract.py            # ⭐ 4-phase polygon extraction
│   ├── stencil_storage.py            # Stencil library (save/load)
│   ├── preprocess.py                 # Face validation
│   ├── yolo_pred.py                  # YOLO detection
│   ├── mediapipe_pred.py             # MediaPipe landmarks
│   ├── utils.py                      # Geometry operations
│   ├── train.py                      # YOLO model training
│   └── start_api.sh                  # API startup script
│
├── frontend/                          # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── upload/UploadPage.jsx     # Image upload + face validation
│   │   │   ├── editor/EditorPage.jsx     # ⭐ Interactive canvas editor
│   │   │   ├── library/LibraryPage.jsx   # Stencil library browser
│   │   │   └── layout/Header.jsx         # Navigation header
│   │   ├── services/apiClient.js     # API wrapper
│   │   └── App.jsx                   # Main app component
│   ├── public/                       # Static assets
│   └── package.json                  # Dependencies
│
├── eyebrow_training/                  # YOLO model
│   └── eyebrow_recommended/weights/best.pt  # Model weights (59MB)
│
├── stencil_data/                      # Stencil library storage
│   ├── stencils.json                 # Master index
│   └── stencil_{uuid}.json           # Individual stencils
│
├── README.md                          # This file (getting started)
├── CLAUDE.md                          # Technical reference (deep dive)
└── requirements.txt                   # Python dependencies
```

---

## ⚙️ Configuration

### Backend Configuration

**Change API settings:**

1. Via API: `POST http://localhost:8000/config`
2. Via code: Edit `stencil_extract.py:DEFAULT_CONFIG`

**Key parameters:**
- `yolo_conf_threshold`: 0.25 (detection confidence)
- `yolo_simplify_epsilon`: 0.005 (polygon simplification)
- `alignment_iou_threshold`: 0.3 (YOLO/MediaPipe agreement)
- `alignment_distance_threshold`: 20.0 (max pixel distance)

### Frontend Configuration

**Change API URL:**

Create `frontend/.env` file:
```
REACT_APP_API_URL=http://your-server:8000
```

Restart React dev server.

---

## 🔧 Troubleshooting

### API Won't Start

**Error**: "YOLO model not loaded"

**Solution**:
```bash
# Verify model exists
ls eyebrow_training/eyebrow_recommended/weights/best.pt

# If missing, pull from Git LFS
git lfs pull

# Reinstall dependencies
pip install ultralytics opencv-python
```

---

### Frontend Shows "Connection Error"

**Error**: "API connection failed"

**Solution**:
```bash
# 1. Verify API is running
curl http://localhost:8000/health

# 2. If not running, start it
./start_api.sh

# 3. Check API URL in frontend
# Should be: http://localhost:8000
cat frontend/src/services/apiClient.js | grep API_BASE_URL
```

---

### Port Already in Use

**Error**: "Address already in use (8000 or 3000)"

**Solution**:
```bash
# Find and kill process on port 8000 (API)
lsof -ti:8000 | xargs kill -9

# Find and kill process on port 3000 (React)
lsof -ti:3000 | xargs kill -9

# Restart services
./start_api.sh
cd frontend && npm start
```

---

### WSL Hot Reload Not Working (React)

**Symptom**: File changes don't appear in browser

**Solution**:
```bash
# Restart React dev server manually
cd frontend
lsof -ti:3000 | xargs kill -9
sleep 2
npm start
```

---

### Image Upload Fails

**Error**: "Face validation failed" or "No eyebrows detected"

**Solutions**:
1. **Check photo requirements**:
   - Front-facing face (not profile)
   - Both eyebrows fully visible
   - Good lighting (not too dark)
   - Not rotated >30° (use straight photo)
   - Hair not covering eyebrows

2. **Try different photo**: Some faces/angles work better than others

3. **Check console logs** (press F12 in browser):
   - Look for specific error messages
   - Check Network tab for API responses

---

## 📊 Performance

### Speed (Typical 800×600 Image)

| Operation | Time |
|-----------|------|
| YOLO detection | 100-150ms |
| MediaPipe detection | 50-100ms |
| Polygon extraction | 200-350ms |
| **Total (upload to editor)** | **400-700ms** |

### Accuracy

- **Alignment success rate**: 75-85% (YOLO + MediaPipe agree)
- **MediaPipe fallback rate**: 15-25% (when misaligned, uses MP only)
- **Polygon point count**: 10-30 points typical

### Resource Usage

- **RAM**: ~200MB (full stack running)
- **CPU**: Single-threaded (no GPU required)
- **Storage**: ~100MB (model + dependencies)

---

## 🤝 Contributing

### Reporting Issues

Found a bug or have a feature request?

1. Check existing issues: [GitHub Issues](https://github.com/your-repo/eyebrow/issues)
2. Create new issue with:
   - Clear description
   - Steps to reproduce
   - Screenshots if applicable
   - System info (OS, Python version, etc.)

### Development Setup

**For contributors:**

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes
4. Test thoroughly
5. Submit pull request

**Code style:**
- Python: Follow PEP 8
- JavaScript: ESLint + Prettier
- Comments: Explain "why" not "what"

---

## 📚 Documentation

### Available Docs

| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** (this file) | Quick start, user guide | End users, beginners |
| **CLAUDE.md** | Technical deep dive, algorithm details | Developers, researchers |
| **API Docs** (http://localhost:8000/docs) | Interactive API reference | API consumers |

### Learning Path

1. **Start here**: README.md for basic usage
2. **Want to integrate?** Check API docs (Swagger UI)
3. **Want to understand internals?** Read CLAUDE.md
4. **Want to customize?** See CLAUDE.md configuration section

---

## 🎯 Use Cases

### Makeup Artists
- Create custom stencils for each client
- Save library of different eyebrow shapes
- Print stencils for use during application

### Beauty Salons
- Offer personalized eyebrow shaping service
- Document client preferences over time
- Demonstrate different eyebrow options

### Product Designers
- Prototype eyebrow stencil templates
- Test different shapes and sizes
- Export designs for manufacturing

### Researchers
- Study facial feature geometry
- Analyze eyebrow symmetry
- Collect datasets for ML training

---

## 📝 License

*To be determined based on project requirements*

---

## 🙏 Acknowledgments

- **YOLO (Ultralytics)** - YOLOv11 segmentation model
- **MediaPipe (Google)** - Facial landmark detection
- **OpenCV** - Image processing and computer vision
- **SciPy** - Scientific computing (polygon operations)
- **FastAPI** - Modern Python web framework
- **React** - UI framework
- **Konva.js** - HTML5 canvas library

---

## 📞 Support

### Getting Help

**Documentation**:
- README (this file) - Quick start and user guide
- CLAUDE.md - Complete technical reference
- API Docs - http://localhost:8000/docs (Swagger UI)

**Community**:
- GitHub Issues - Report bugs and request features
- Discussions - Ask questions and share tips

**Logs**:
- Backend: Terminal output where API is running
- Frontend: Browser console (press F12)
- Stencil data: `stencil_data/` directory

---

## 🔑 Quick Commands Reference

### Startup

```bash
# Terminal 1: Start API
./start_api.sh

# Terminal 2: Start frontend
cd frontend && npm start

# Open browser
open http://localhost:3000
```

### Health Check

```bash
# Check API
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Stop Services

```bash
# Stop API (press Ctrl+C in API terminal)
# OR
pkill -f uvicorn

# Stop frontend (press Ctrl+C in frontend terminal)
```

---

## 🚢 Deployment

### Production Build

**Backend**:
```bash
# Use production ASGI server (e.g., Gunicorn)
gunicorn api.api_main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend**:
```bash
cd frontend
npm run build
# Serve build/ directory with Nginx or Apache
```

### Docker (Future)

Docker support planned for easier deployment.

---

## 📈 Roadmap

### v6.0 (Current) ✅
- 4-phase polygon extraction ("grounding" algorithm)
- React frontend with interactive canvas editor
- Stencil library system (save, list, delete)
- 19 REST API endpoints

### v6.1 (Planned)
- Export to SVG/PNG with sizing guides
- Print-ready stencil generation
- Advanced editing tools (smoothing, symmetry matching)

### v7.0 (Future)
- Batch processing (multiple images)
- Mobile app (React Native)
- User authentication (accounts, sharing)
- Cloud deployment (Docker + Kubernetes)

---

**Project Status: ✅ PRODUCTION READY**

Complete end-to-end stencil creation system. Ready to use!

---

*Last Updated: 2025-01-13*
*Version: 6.0*
*Made with ❤️ for the beauty industry*
