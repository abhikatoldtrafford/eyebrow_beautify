# Brow Stencil Frontend

**React-based interactive editor for custom eyebrow stencils**

*Built with React 19, Konva.js, and React Router*

---

## 🎨 Overview

The Brow Stencil Frontend is a modern React application that provides an interactive canvas-based editor for creating and editing custom eyebrow stencils. Users can upload photos, edit polygon boundaries with precision zoom and pan controls, save stencils to a library, and manage their collection.

**Key Features:**
- Interactive canvas editing with Konva.js
- Zoom (0.5x-5.0x) and pan controls
- Drag-and-drop control point editing
- Add/delete vertices by double-click or Delete key
- Save to library with metadata
- Browse and filter saved stencils
- Responsive design for desktop and tablet

---

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm 8+
- Backend API running on `http://localhost:8000`

### Installation

```bash
# From the eyebrow directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The app will open at **http://localhost:3000**

---

## 📦 Dependencies

### Core Technologies

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | 19.2.0 | UI framework |
| `react-dom` | 19.2.0 | React DOM rendering |
| `react-router-dom` | 7.9.5 | Client-side routing (3 pages) |

### Canvas & Editing

| Package | Version | Purpose |
|---------|---------|---------|
| `konva` | 10.0.8 | HTML5 canvas library (zoom, pan, transforms) |
| `react-konva` | 19.2.0 | React bindings for Konva |
| `use-image` | 1.1.4 | Image loading hook for Konva |

### API & File Handling

| Package | Version | Purpose |
|---------|---------|---------|
| `axios` | 1.13.2 | HTTP client for API calls |
| `react-dropzone` | 14.3.8 | Drag-and-drop file upload |

### Development Tools

| Package | Version | Purpose |
|---------|---------|---------|
| `react-scripts` | 5.0.1 | Create React App build tooling |
| `@testing-library/react` | 16.3.0 | Component testing |
| `@testing-library/jest-dom` | 6.9.1 | Jest matchers for DOM |

---

## 🗂️ Project Structure

```
frontend/
├── public/
│   ├── index.html               # HTML template
│   ├── favicon.ico              # App icon
│   └── manifest.json            # PWA manifest
│
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   └── Header.jsx       # Navigation header (3 routes)
│   │   │
│   │   ├── upload/
│   │   │   ├── UploadPage.jsx   # Image upload + face validation
│   │   │   └── UploadPage.css
│   │   │
│   │   ├── editor/
│   │   │   ├── EditorPage.jsx   # ⭐ Interactive canvas editor (Konva)
│   │   │   └── EditorPage.css
│   │   │
│   │   └── library/
│   │       ├── LibraryPage.jsx  # Stencil library browser
│   │       └── LibraryPage.css
│   │
│   ├── services/
│   │   └── apiClient.js         # API wrapper (8 functions)
│   │
│   ├── App.jsx                  # Main app component (routing)
│   ├── App.css                  # Global styles
│   ├── index.js                 # React entry point
│   └── index.css                # Base CSS
│
├── package.json                 # Dependencies
├── package-lock.json            # Locked dependency versions
└── README.md                    # This file
```

**Total Files:** 11 JSX/JS files (~1,200 lines)

---

## 🧩 Component Architecture

### 1. App.jsx (Main Router)

```jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/layout/Header';
import UploadPage from './components/upload/UploadPage';
import EditorPage from './components/editor/EditorPage';
import LibraryPage from './components/library/LibraryPage';

function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/editor" element={<EditorPage />} />
        <Route path="/library" element={<LibraryPage />} />
      </Routes>
    </BrowserRouter>
  );
}
```

**3 Routes:**
- `/` → Upload page (drag-and-drop image upload)
- `/editor` → Canvas editor (interactive polygon editing)
- `/library` → Stencil library (browse, filter, delete)

---

### 2. UploadPage.jsx (Image Upload)

**Features:**
- Drag-and-drop file upload (React Dropzone)
- Accepts JPG/PNG (max 10MB)
- Image preview
- Calls API `/beautify/base64` endpoint
- Face validation with preprocessing
- Automatic navigation to Editor on success

**API Flow:**
```
User drops image
  ↓
Convert to Base64
  ↓
POST /beautify/base64
  ↓
Receive polygon points (10-30 vertices)
  ↓
Navigate to /editor with state:
  - image (data URL)
  - imageBase64 (for saving)
  - stencils (left/right polygons)
```

**Key Functions:**
- `onDrop()` - Handle file upload
- `handleExtract()` - Call API and navigate to editor
- `fileToBase64()` - Convert image to Base64

---

### 3. EditorPage.jsx (Interactive Canvas) ⭐

**The Core Component** - 574 lines of interactive canvas logic

#### Konva Stage Structure

```jsx
<Stage
  width={800}
  height={600}
  onWheel={handleWheel}  // Zoom control
>
  <Layer
    scaleX={imageScale * userZoom}
    scaleY={imageScale * userZoom}
    x={panPosition.x}
    y={panPosition.y}
    draggable={true}      // Pan control
    onDragEnd={handleDragEnd}
  >
    {/* Background image */}
    <KonvaImage image={imageObj} />

    {/* Left eyebrow polygon (green) */}
    <EditablePolygon
      points={leftPolygon}
      color="rgba(0, 255, 0, 0.3)"
      onPointDrag={handleLeftPointDrag}
      onLineDoubleClick={handleLeftLineDoubleClick}
    />

    {/* Right eyebrow polygon (red) */}
    <EditablePolygon
      points={rightPolygon}
      color="rgba(255, 0, 0, 0.3)"
      onPointDrag={handleRightPointDrag}
      onLineDoubleClick={handleRightLineDoubleClick}
    />
  </Layer>
</Stage>
```

#### Interactive Controls

| Control | Action | Implementation |
|---------|--------|----------------|
| **Zoom** | Mouse wheel | `onWheel` → adjust `userZoom` (0.5x-5.0x) |
| **Pan** | Drag canvas | `Layer draggable={true}` → update `panPosition` |
| **Move point** | Drag white circle | `Circle` with `draggable={true}` → update polygon |
| **Add point** | Double-click edge | `Line` with `onDblClick` → insert vertex at click position |
| **Delete point** | Click + Delete key | `useEffect` keyboard listener → remove vertex from array |
| **Reset view** | Click button | Set `userZoom=1`, `panPosition={x:0, y:0}` |

#### Zoom Implementation

```jsx
const handleWheel = (e) => {
  e.evt.preventDefault();

  const scaleBy = 1.05;
  const oldZoom = userZoom;

  // Zoom in/out
  const newZoom = e.evt.deltaY > 0
    ? oldZoom / scaleBy  // Zoom out
    : oldZoom * scaleBy; // Zoom in

  // Clamp zoom range
  setUserZoom(Math.max(0.5, Math.min(5.0, newZoom)));
};
```

#### Pan Implementation

```jsx
// Layer is draggable, so panning is automatic
// Just need to track position for reset
const handleDragEnd = (e) => {
  setPanPosition({
    x: e.target.x(),
    y: e.target.y()
  });
};
```

#### Point Dragging

```jsx
const handleLeftPointDrag = (index, e) => {
  // Get new position in canvas coordinates
  const newX = (e.target.x() - panPosition.x) / (imageScale * userZoom);
  const newY = (e.target.y() - panPosition.y) / (imageScale * userZoom);

  // Update polygon array
  const newPolygon = [...leftPolygon];
  newPolygon[index] = [newX, newY];
  setLeftPolygon(newPolygon);
};
```

#### Adding Vertices (Double-Click)

```jsx
const handleLeftLineDoubleClick = (index, e) => {
  // Get click position
  const clickX = (e.evt.layerX - panPosition.x) / (imageScale * userZoom);
  const clickY = (e.evt.layerY - panPosition.y) / (imageScale * userZoom);

  // Insert new vertex between index and index+1
  const newPolygon = [...leftPolygon];
  newPolygon.splice(index + 1, 0, [clickX, clickY]);
  setLeftPolygon(newPolygon);
};
```

#### Deleting Vertices (Delete Key)

```jsx
useEffect(() => {
  const handleKeyDown = (e) => {
    if (e.key === 'Delete' && selectedPointIndex !== null) {
      // Remove vertex at selectedPointIndex
      if (selectedSide === 'left') {
        setLeftPolygon(prev => prev.filter((_, i) => i !== selectedPointIndex));
      } else {
        setRightPolygon(prev => prev.filter((_, i) => i !== selectedPointIndex));
      }
      setSelectedPointIndex(null);
    }
  };

  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [selectedPointIndex, selectedSide]);
```

#### Save to Library

```jsx
const handleSaveStencil = async () => {
  try {
    // Save left eyebrow
    await apiClient.saveStencil({
      polygon: leftPolygon,
      side: 'left',
      image_base64: imageBase64,
      metadata: { tags: [], notes: '' }
    });

    // Save right eyebrow
    await apiClient.saveStencil({
      polygon: rightPolygon,
      side: 'right',
      image_base64: imageBase64,
      metadata: { tags: [], notes: '' }
    });

    alert('Stencils saved to library!');
    navigate('/library');
  } catch (error) {
    console.error('Save failed:', error);
  }
};
```

---

### 4. LibraryPage.jsx (Stencil Browser)

**Features:**
- List all saved stencils (GET `/stencils/list`)
- Filter by side (left/right tabs)
- Display thumbnail previews
- Delete stencils (DELETE `/stencils/{id}`)
- Show metadata (created date, tags, point count)

**API Flow:**
```
Component mounts
  ↓
GET /stencils/list
  ↓
Display as cards
  ↓
User clicks "Delete"
  ↓
DELETE /stencils/{id}
  ↓
Refresh list
```

---

### 5. apiClient.js (API Service Layer)

**8 API Functions:**

```javascript
// Base URL configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default {
  // Health check
  checkHealth: async () => {...},

  // Face preprocessing (validation + rotation detection)
  preprocess: async (imageBase64) => {
    return axios.post(`${API_BASE_URL}/preprocess`, {
      image_base64: imageBase64
    });
  },

  // Main stencil extraction (4-phase polygon extraction)
  extractStencil: async (imageBase64) => {
    return axios.post(`${API_BASE_URL}/beautify/base64`, {
      image_base64: imageBase64
    });
  },

  // Save stencil to library
  saveStencil: async (data) => {
    return axios.post(`${API_BASE_URL}/stencils/save`, data);
  },

  // List all stencils (with optional filtering)
  listStencils: async (side = null) => {
    const url = side
      ? `${API_BASE_URL}/stencils/list?side=${side}`
      : `${API_BASE_URL}/stencils/list`;
    return axios.get(url);
  },

  // Get specific stencil by ID
  getStencil: async (stencilId) => {
    return axios.get(`${API_BASE_URL}/stencils/${stencilId}`);
  },

  // Delete stencil from library
  deleteStencil: async (stencilId) => {
    return axios.delete(`${API_BASE_URL}/stencils/${stencilId}`);
  },

  // Convert file to Base64
  fileToBase64: (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
};
```

---

## 🎨 Styling

### CSS Organization

| File | Purpose |
|------|---------|
| `index.css` | Global base styles, CSS reset |
| `App.css` | App-level layout, common utilities |
| `UploadPage.css` | Upload page specific styles (dropzone, buttons) |
| `EditorPage.css` | Editor page specific styles (canvas, controls, instructions) |
| `LibraryPage.css` | Library page specific styles (cards, filters, grid) |

### Design System

**Colors:**
- Primary: `#4A90E2` (blue)
- Success: `#4CAF50` (green)
- Error: `#F44336` (red)
- Left Eyebrow: `rgba(0, 255, 0, 0.3)` (transparent green)
- Right Eyebrow: `rgba(255, 0, 0, 0.3)` (transparent red)
- Control Points: `white` circles with `black` stroke

**Typography:**
- Font Family: `-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'`
- Base Size: `16px`
- Headings: `1.5rem - 2rem`

**Spacing:**
- Base Unit: `8px`
- Standard Padding: `16px`
- Component Margin: `24px`

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in `frontend/` directory:

```bash
# API Base URL
REACT_APP_API_URL=http://localhost:8000

# Optional: Enable debug mode
REACT_APP_DEBUG=false
```

### API URL Configuration

**Default:** `http://localhost:8000`

**Override for production:**
```bash
# .env.production
REACT_APP_API_URL=https://your-api-domain.com
```

**Or set at build time:**
```bash
REACT_APP_API_URL=https://api.example.com npm run build
```

---

## 🛠️ Development Workflow

### Start Development Server

```bash
npm start
```

**Runs on:** http://localhost:3000
**Features:**
- Hot module reloading
- Automatic browser refresh
- Error overlay in browser

### Build for Production

```bash
npm run build
```

**Output:** `build/` directory
**Optimizations:**
- Minified code
- Tree-shaking (remove unused code)
- Code splitting
- Asset hashing for cache busting

### Serve Production Build Locally

```bash
# Install serve globally
npm install -g serve

# Serve build directory
serve -s build -l 3000
```

### Run Tests

```bash
npm test
```

**Test Libraries:**
- Jest (test runner)
- React Testing Library (component testing)
- jsdom (DOM simulation)

---

## 🐛 Troubleshooting

### API Connection Error

**Symptom:** "Failed to fetch" or "Network Error" messages

**Solution:**
```bash
# Verify API is running
curl http://localhost:8000/health

# If not running, start it
cd /mnt/g/eyebrow
./start_api.sh

# Check API URL in frontend
cat frontend/.env
# Should show: REACT_APP_API_URL=http://localhost:8000
```

---

### Port 3000 Already in Use

**Symptom:** "Port 3000 is already in use"

**Solution:**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Restart frontend
npm start
```

---

### Hot Reload Not Working (WSL)

**Symptom:** File changes don't trigger browser refresh in WSL

**Solution:**
```bash
# Use polling mode
CHOKIDAR_USEPOLLING=true npm start

# Or add to .env
echo "CHOKIDAR_USEPOLLING=true" >> .env
```

---

### Canvas Not Rendering Image

**Symptom:** Blank canvas or missing image

**Solution:**
- Check browser console for CORS errors
- Verify image was passed via router state
- Check `useImage` hook loaded successfully
- Ensure image is Base64 or valid URL

**Debug:**
```jsx
// Add to EditorPage.jsx
useEffect(() => {
  console.log('Image state:', location.state?.image);
  console.log('Stencils:', location.state?.stencils);
}, [location.state]);
```

---

### Zoom/Pan Not Working

**Symptom:** Can't zoom with mouse wheel or pan canvas

**Solution:**
- Check `onWheel` handler is attached to `<Stage>`
- Verify `Layer` has `draggable={true}`
- Ensure `e.evt.preventDefault()` in `handleWheel`
- Check zoom limits (0.5x-5.0x)

---

### Points Not Draggable

**Symptom:** Control points (white circles) won't move

**Solution:**
- Check `Circle` has `draggable={true}`
- Verify `onDragMove` handler attached
- Ensure `e.cancelBubble = true` to prevent Layer drag
- Check coordinate transformation math

---

## 📊 Performance Optimization

### Best Practices

1. **Memoize Expensive Calculations**
```jsx
const transformedPoints = useMemo(() => {
  return leftPolygon.map(([x, y]) => [
    x * imageScale * userZoom + panPosition.x,
    y * imageScale * userZoom + panPosition.y
  ]);
}, [leftPolygon, imageScale, userZoom, panPosition]);
```

2. **Debounce API Calls**
```jsx
import { debounce } from 'lodash';

const debouncedSave = useMemo(
  () => debounce(saveStencil, 500),
  []
);
```

3. **Lazy Load Routes**
```jsx
import { lazy, Suspense } from 'react';

const EditorPage = lazy(() => import('./components/editor/EditorPage'));
const LibraryPage = lazy(() => import('./components/library/LibraryPage'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Routes>
        <Route path="/editor" element={<EditorPage />} />
        <Route path="/library" element={<LibraryPage />} />
      </Routes>
    </Suspense>
  );
}
```

4. **Optimize Konva Rendering**
```jsx
// Use listening: false for static shapes
<Circle listening={false} />

// Use cache for complex shapes
<Line
  perfectDrawEnabled={false}
  shadowForStrokeEnabled={false}
/>
```

---

## 🚀 Deployment

### Static Hosting (Netlify, Vercel)

```bash
# Build for production
npm run build

# Deploy build/ directory to host
# Update REACT_APP_API_URL to production API
```

**Netlify:**
```toml
# netlify.toml
[build]
  command = "npm run build"
  publish = "build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

**Vercel:**
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### Docker

```dockerfile
FROM node:16-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## 📚 Additional Resources

- **Main README**: `../README.md` (User guide)
- **Technical Reference**: `../CLAUDE.md` (Algorithm deep dive)
- **API Documentation**: `../api/README.md` (API reference)
- **React Docs**: https://react.dev
- **Konva Docs**: https://konvajs.org/docs/
- **React Konva**: https://konvajs.org/docs/react/

---

## 🎯 Architecture Summary

```
User Flow:
  1. Upload Page → Drop image
     ↓
  2. API Call → POST /beautify/base64
     ↓
  3. Editor Page → Interactive canvas (Konva)
     - Zoom: Mouse wheel (0.5x-5.0x)
     - Pan: Drag canvas
     - Edit: Drag control points
     - Add: Double-click edge
     - Delete: Select + Delete key
     ↓
  4. Save → POST /stencils/save (left + right)
     ↓
  5. Library Page → Browse, filter, delete
```

**Technology Stack:**
- React 19.2 (UI framework)
- Konva 10.0.8 (Canvas rendering)
- React Router 7.9.5 (Navigation)
- Axios 1.13.2 (API client)

**Key Files:**
- `EditorPage.jsx` (574 lines - core interactive canvas)
- `apiClient.js` (API service layer)
- `UploadPage.jsx` (Image upload + validation)
- `LibraryPage.jsx` (Stencil management)

---

**Frontend Version: 6.0 | Production Ready** 🚀

Start the development server with `npm start` and visit http://localhost:3000 to begin!
