import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Stage, Layer, Image as KonvaImage, Line, Circle, Text } from 'react-konva';
import useImage from 'use-image';
import apiClient from '../../services/apiClient';
import './EditorPage.css';

// Component to render background image
const BackgroundImage = ({ src, onLoad }) => {
  console.log('BackgroundImage - src type:', typeof src, 'length:', src?.length);
  const [image] = useImage(src);

  React.useEffect(() => {
    console.log('BackgroundImage - image loaded:', !!image, 'dimensions:', image?.width, 'x', image?.height);
    if (image && onLoad) {
      onLoad(image.width, image.height);
    }
  }, [image, onLoad]);

  if (!image) {
    console.log('BackgroundImage - image not loaded yet, returning null');
    return null;
  }

  console.log('BackgroundImage - rendering KonvaImage');
  return (
    <KonvaImage
      image={image}
    />
  );
};

// Component to render editable polygon with smooth curves
const EditablePolygon = ({ points, onPointsChange, color, side }) => {
  const [selectedPoint, setSelectedPoint] = useState(null);

  const handleDragMove = (index, e) => {
    e.cancelBubble = true; // Prevent Layer from dragging
    const newPoints = [...points];
    newPoints[index] = [e.target.x(), e.target.y()];
    onPointsChange(newPoints);
  };

  const handlePointClick = (index, e) => {
    if (e.evt.button === 2) { // Right-click
      e.evt.preventDefault();
      if (points.length > 5) { // Minimum 5 points
        const newPoints = points.filter((_, i) => i !== index);
        onPointsChange(newPoints);
      }
    } else {
      setSelectedPoint(index);
    }
  };

  const handleLineClick = (e) => {
    // Add new point when clicking on the curve
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    // Find closest edge to insert point
    let minDist = Infinity;
    let insertIndex = 0;

    for (let i = 0; i < points.length; i++) {
      const p1 = points[i];
      const p2 = points[(i + 1) % points.length];
      const dist = distanceToSegment(point, p1, p2);
      if (dist < minDist) {
        minDist = dist;
        insertIndex = i + 1;
      }
    }

    if (minDist < 20) { // Only add if close to curve
      const newPoints = [...points];
      newPoints.splice(insertIndex, 0, [point.x, point.y]);
      onPointsChange(newPoints);
    }
  };

  // Flatten points for Konva Line component
  const flatPoints = points.flat();

  return (
    <>
      {/* Smooth curve using bezier */}
      <Line
        points={flatPoints}
        stroke={color}
        strokeWidth={3}
        closed
        tension={0.4} // This creates smooth curves!
        onClick={handleLineClick}
        listening
      />

      {/* Control points */}
      {points.map((point, index) => (
        <Circle
          key={index}
          x={point[0]}
          y={point[1]}
          radius={selectedPoint === index ? 8 : 6}
          fill={selectedPoint === index ? '#ff6600' : '#ffaa00'}
          stroke="#ff6600"
          strokeWidth={2}
          draggable
          onDragStart={(e) => { e.cancelBubble = true; }}
          onDragMove={(e) => handleDragMove(index, e)}
          onDragEnd={(e) => { e.cancelBubble = true; }}
          onClick={(e) => handlePointClick(index, e)}
          onMouseEnter={(e) => {
            const container = e.target.getStage().container();
            container.style.cursor = 'pointer';
          }}
          onMouseLeave={(e) => {
            const container = e.target.getStage().container();
            container.style.cursor = 'default';
          }}
        />
      ))}

      {/* Label */}
      <Text
        x={points[0][0] - 30}
        y={points[0][1] - 30}
        text={side.toUpperCase()}
        fontSize={14}
        fill={color}
        fontStyle="bold"
      />
    </>
  );
};

// Distance from point to line segment
const distanceToSegment = (point, p1, p2) => {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return Math.sqrt((point.x - p1[0]) ** 2 + (point.y - p1[1]) ** 2);

  const t = Math.max(0, Math.min(1, ((point.x - p1[0]) * dx + (point.y - p1[1]) * dy) / (len * len)));
  const projX = p1[0] + t * dx;
  const projY = p1[1] + t * dy;

  return Math.sqrt((point.x - projX) ** 2 + (point.y - projY) ** 2);
};

const EditorPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  console.log('EditorPage loaded - location.state:', location.state);

  const { image, imageBase64, stencils } = location.state || {};

  console.log('Extracted props:', {
    hasImage: !!image,
    imageType: typeof image,
    imageLength: image?.length,
    hasImageBase64: !!imageBase64,
    hasStencils: !!stencils,
    stencilsCount: stencils?.length
  });

  // Debugging alert
  if (!location.state) {
    alert('ERROR: location.state is null/undefined!');
  }

  const [leftPolygon, setLeftPolygon] = useState(null);
  const [rightPolygon, setRightPolygon] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 });
  const [imageScale, setImageScale] = useState(1);
  const [userZoom, setUserZoom] = useState(1);
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  // Handle image load to get proper dimensions
  const handleImageLoad = useCallback((imgWidth, imgHeight) => {
    console.log('handleImageLoad called with:', imgWidth, 'x', imgHeight);
    const maxWidth = 900;
    const maxHeight = 700;

    let width = imgWidth;
    let height = imgHeight;
    let scale = 1;

    // Scale down if too large while preserving aspect ratio
    if (width > maxWidth || height > maxHeight) {
      scale = Math.min(maxWidth / width, maxHeight / height);
      console.log('Scaling ratio:', scale);
      width = width * scale;
      height = height * scale;
    }

    console.log('Setting canvas size to:', width, 'x', height);
    console.log('Setting image scale to:', scale);
    setCanvasSize({ width, height });
    setImageScale(scale);
  }, []); // Empty deps - this function doesn't depend on any props or state

  useEffect(() => {
    if (!stencils || stencils.length === 0) {
      console.log('No stencils found, redirecting to upload');
      navigate('/');
      return;
    }

    console.log('Initializing polygons from stencils:', stencils);

    // Initialize polygons - store entire stencil object
    stencils.forEach(stencil => {
      console.log(`Processing stencil - side: ${stencil.side}, points:`, stencil.polygon?.length || 0);
      if (stencil.side === 'left' && stencil.polygon && stencil.polygon.length > 0) {
        setLeftPolygon({ ...stencil, polygon: stencil.polygon });
        console.log('Left polygon set with', stencil.polygon.length, 'points');
      } else if (stencil.side === 'right' && stencil.polygon && stencil.polygon.length > 0) {
        setRightPolygon({ ...stencil, polygon: stencil.polygon });
        console.log('Right polygon set with', stencil.polygon.length, 'points');
      }
    });
  }, [stencils, navigate]);

  // Initialize history when polygons are loaded
  useEffect(() => {
    if ((leftPolygon || rightPolygon) && history.length === 0) {
      const state = {
        left: leftPolygon ? JSON.parse(JSON.stringify(leftPolygon)) : null,
        right: rightPolygon ? JSON.parse(JSON.stringify(rightPolygon)) : null
      };
      setHistory([state]);
      setHistoryIndex(0);
    }
  }, [leftPolygon, rightPolygon, history.length]);

  const saveHistory = () => {
    const state = {
      left: leftPolygon ? JSON.parse(JSON.stringify(leftPolygon)) : null,
      right: rightPolygon ? JSON.parse(JSON.stringify(rightPolygon)) : null
    };
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(state);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleLeftChange = (newPoints) => {
    setLeftPolygon(prev => ({ ...prev, polygon: newPoints }));
    saveHistory();
  };

  const handleRightChange = (newPoints) => {
    setRightPolygon(prev => ({ ...prev, polygon: newPoints }));
    saveHistory();
  };

  const undo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      const state = history[newIndex];
      setLeftPolygon(state.left);
      setRightPolygon(state.right);
      setHistoryIndex(newIndex);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      const state = history[newIndex];
      setLeftPolygon(state.left);
      setRightPolygon(state.right);
      setHistoryIndex(newIndex);
    }
  };

  const handleZoomIn = () => {
    setUserZoom(prev => Math.min(prev * 1.2, 5)); // Max 5x zoom
  };

  const handleZoomOut = () => {
    setUserZoom(prev => Math.max(prev / 1.2, 0.5)); // Min 0.5x zoom
  };

  const handleFitToView = () => {
    setUserZoom(1);
    setPanPosition({ x: 0, y: 0 });
  };

  const handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.1;
    const stage = e.target.getStage();
    const oldScale = userZoom;
    const pointer = stage.getPointerPosition();

    const newScale = e.evt.deltaY > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    const clampedScale = Math.max(0.5, Math.min(5, newScale));

    // Adjust pan to zoom towards mouse position
    const mousePointTo = {
      x: (pointer.x - panPosition.x) / oldScale,
      y: (pointer.y - panPosition.y) / oldScale,
    };

    const newPos = {
      x: pointer.x - mousePointTo.x * clampedScale,
      y: pointer.y - mousePointTo.y * clampedScale,
    };

    setUserZoom(clampedScale);
    setPanPosition(newPos);
  };

  const handleDragStart = (e) => {
    // Only allow panning if not dragging a shape (Circle, Line)
    const target = e.target;
    if (target.getClassName() === 'Layer') {
      setIsDragging(true);
    }
  };

  const handleDragEnd = (e) => {
    const target = e.target;
    // Only update pan position if we were dragging the Layer (not shapes)
    if (target.getClassName() === 'Layer') {
      setPanPosition({
        x: target.x(),
        y: target.y(),
      });
    }
    setIsDragging(false);
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);

    try {
      // Helper function to round polygon coordinates to integers
      const roundPolygon = (polygon) => {
        return polygon.map(point => [
          Math.round(point[0]),
          Math.round(point[1])
        ]);
      };

      const stencilsToSave = [];

      if (leftPolygon) {
        stencilsToSave.push({
          polygon: roundPolygon(leftPolygon.polygon),
          side: 'left',
          name: `Left Brow ${new Date().toLocaleDateString()}`,
          tags: ['auto-generated'],
          notes: 'Created with Brow Stencil App',
          image_base64: imageBase64
        });
      }

      if (rightPolygon) {
        stencilsToSave.push({
          polygon: roundPolygon(rightPolygon.polygon),
          side: 'right',
          name: `Right Brow ${new Date().toLocaleDateString()}`,
          tags: ['auto-generated'],
          notes: 'Created with Brow Stencil App',
          image_base64: imageBase64
        });
      }

      for (const stencil of stencilsToSave) {
        await apiClient.saveStencil(stencil);
      }

      setMessage({ type: 'success', text: 'Stencils saved successfully!' });

      setTimeout(() => {
        navigate('/library');
      }, 1500);

    } catch (err) {
      console.error('Save error:', err);

      // Handle different error formats
      let errorText = 'Failed to save stencils';
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        // If detail is an array of validation errors
        if (Array.isArray(detail)) {
          errorText = detail.map(e => e.msg || JSON.stringify(e)).join(', ');
        }
        // If detail is an object
        else if (typeof detail === 'object') {
          errorText = detail.msg || JSON.stringify(detail);
        }
        // If detail is a string
        else {
          errorText = detail;
        }
      } else if (err.message) {
        errorText = err.message;
      }

      setMessage({
        type: 'error',
        text: errorText
      });
      setSaving(false);
    }
  };

  if (!image) {
    console.error('No image found in location.state');
    return (
      <div className="editor-page">
        <div className="error-message">
          No image data found. Please upload an image first.
        </div>
        <button onClick={() => navigate('/')} className="btn btn-primary">
          Back to Upload
        </button>
      </div>
    );
  }

  console.log('EditorPage render:', {
    hasImage: !!image,
    hasStencils: !!stencils,
    stencilCount: stencils?.length,
    hasLeftPolygon: !!leftPolygon,
    hasRightPolygon: !!rightPolygon,
    canvasSize
  });

  return (
    <div className="editor-page">
      <div className="editor-header">
        <h2>Edit Your Stencil</h2>
        <p>Drag control points to adjust the shape. Click on curve to add points. Right-click points to remove.</p>
      </div>

      {message && (
        <div className={`${message.type}-message`}>
          {message.text}
        </div>
      )}

      <div className="editor-controls">
        <button
          onClick={undo}
          disabled={historyIndex <= 0}
          className="btn btn-secondary"
        >
          ← Undo
        </button>

        <button
          onClick={redo}
          disabled={historyIndex >= history.length - 1}
          className="btn btn-secondary"
        >
          Redo →
        </button>

        <div className="spacer"></div>

        <button
          onClick={handleZoomOut}
          className="btn btn-secondary"
        >
          🔍- Zoom Out
        </button>

        <button
          onClick={handleFitToView}
          className="btn btn-secondary"
        >
          ⊡ Fit to View
        </button>

        <button
          onClick={handleZoomIn}
          className="btn btn-secondary"
        >
          🔍+ Zoom In
        </button>

        <div className="spacer"></div>

        <button
          onClick={() => navigate('/')}
          className="btn btn-secondary"
        >
          Cancel
        </button>

        <button
          onClick={handleSave}
          disabled={saving}
          className="btn btn-primary"
        >
          {saving ? (
            <>
              <span className="loading-spinner"></span>
              Saving...
            </>
          ) : (
            '💾 Save Stencil'
          )}
        </button>
      </div>

      <div className="editor-canvas">
        <Stage
          width={canvasSize.width}
          height={canvasSize.height}
          onWheel={handleWheel}
        >
          <Layer
            scaleX={imageScale * userZoom}
            scaleY={imageScale * userZoom}
            x={panPosition.x}
            y={panPosition.y}
            draggable={true}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            <BackgroundImage src={image} onLoad={handleImageLoad} />

            {leftPolygon && leftPolygon.polygon && (
              <EditablePolygon
                points={leftPolygon.polygon}
                onPointsChange={handleLeftChange}
                color="#00ff00"
                side="left"
              />
            )}

            {rightPolygon && rightPolygon.polygon && (
              <EditablePolygon
                points={rightPolygon.polygon}
                onPointsChange={handleRightChange}
                color="#00ffff"
                side="right"
              />
            )}
          </Layer>
        </Stage>
      </div>

      <div className="editor-info card">
        <h3>🎯 Editing Instructions</h3>
        <ul>
          <li><strong>Zoom:</strong> Use mouse wheel or zoom buttons to zoom in/out</li>
          <li><strong>Pan:</strong> Click and drag canvas to move around</li>
          <li><strong>Drag points:</strong> Click and drag orange circles to reshape curve</li>
          <li><strong>Add point:</strong> Click on green/cyan curve</li>
          <li><strong>Remove point:</strong> Right-click on orange circle</li>
          <li><strong>Undo/Redo:</strong> Use buttons above or Ctrl+Z / Ctrl+Shift+Z</li>
        </ul>
      </div>
    </div>
  );
};

export default EditorPage;
