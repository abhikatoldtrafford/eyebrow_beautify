import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import apiClient from '../../services/apiClient';
import './UploadPage.css';

const UploadPage = () => {
  const navigate = useNavigate();
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('');

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setError(null);
    setStatus('');

    // Create preview
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);

    // Convert to base64 for API
    const base64 = await apiClient.fileToBase64(file);
    setImage(base64);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    multiple: false,
    maxSize: 10485760 // 10MB
  });

  const handleExtract = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      // Step 1: Preprocessing
      setStatus('Validating face...');
      const prepResult = await apiClient.preprocess(image);

      if (!prepResult.valid) {
        setError(`Face validation failed: ${prepResult.rejection_reason || 'Invalid face detected'}`);
        setLoading(false);
        return;
      }

      // Step 2: Extract stencil polygons
      setStatus('Extracting stencil polygons...');
      const extractResult = await apiClient.extractStencil(image);

      if (!extractResult.success || extractResult.stencils.length === 0) {
        setError('No eyebrows detected. Please try another image.');
        setLoading(false);
        return;
      }

      // Navigate to editor with results
      console.log('Navigating to editor with state:', {
        hasImagePreview: !!imagePreview,
        imagePreviewType: typeof imagePreview,
        imagePreviewLength: imagePreview?.length,
        hasImageBase64: !!image,
        stencilsCount: extractResult.stencils?.length
      });

      navigate('/editor', {
        state: {
          image: imagePreview,
          imageBase64: image,
          stencils: extractResult.stencils
        }
      });

    } catch (err) {
      console.error('Extraction error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to extract stencil');
      setLoading(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-header">
        <h2>Upload Your Photo</h2>
        <p>Upload a front-facing photo to generate your custom brow stencil</p>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="upload-container">
        {!imagePreview ? (
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-content">
              <div className="upload-icon">📸</div>
              {isDragActive ? (
                <p>Drop the image here...</p>
              ) : (
                <>
                  <p className="dropzone-text">
                    Drag & drop an image here, or click to select
                  </p>
                  <p className="dropzone-hint">
                    Supports: JPG, PNG (max 10MB)
                  </p>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="image-preview-container">
            <div className="image-preview">
              <img src={imagePreview} alt="Uploaded" />
            </div>

            <div className="upload-actions">
              <button
                onClick={() => {
                  setImage(null);
                  setImagePreview(null);
                  setError(null);
                  setStatus('');
                }}
                className="btn btn-secondary"
                disabled={loading}
              >
                Choose Different Image
              </button>

              <button
                onClick={handleExtract}
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    {status || 'Processing...'}
                  </>
                ) : (
                  'Extract Stencil'
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="upload-tips card">
        <h3>📝 Tips for Best Results</h3>
        <ul>
          <li>Use a front-facing photo with good lighting</li>
          <li>Ensure both eyebrows are fully visible</li>
          <li>Avoid extreme tilting (max 30 degrees)</li>
          <li>Remove glasses if possible</li>
          <li>Hair should not cover eyebrows</li>
        </ul>
      </div>
    </div>
  );
};

export default UploadPage;
