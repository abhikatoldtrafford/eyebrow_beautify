import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import apiClient from '../../services/apiClient';
import './LibraryPage.css';

const StencilCard = ({ stencil, onDelete, onExport }) => {
  return (
    <div className="stencil-card">
      <div className="stencil-card-header">
        <span className={`side-badge ${stencil.side}`}>
          {stencil.side.toUpperCase()}
        </span>
        <span className="stencil-points">{stencil.num_points} pts</span>
      </div>

      <div className="stencil-info">
        <h3>{stencil.name || 'Untitled Stencil'}</h3>
        <p className="stencil-date">
          {new Date(stencil.created_at).toLocaleDateString()}
        </p>
        {stencil.tags && stencil.tags.length > 0 && (
          <div className="stencil-tags">
            {stencil.tags.map((tag, i) => (
              <span key={i} className="tag">{tag}</span>
            ))}
          </div>
        )}
      </div>

      <div className="stencil-actions">
        <button
          onClick={() => onExport(stencil.stencil_id, 'svg')}
          className="btn-icon"
          title="Export as SVG"
        >
          📥 SVG
        </button>
        <button
          onClick={() => onExport(stencil.stencil_id, 'json')}
          className="btn-icon"
          title="Export as JSON"
        >
          📄 JSON
        </button>
        <button
          onClick={() => onDelete(stencil.stencil_id)}
          className="btn-icon btn-delete"
          title="Delete"
        >
          🗑️
        </button>
      </div>
    </div>
  );
};

const LibraryPage = () => {
  const navigate = useNavigate();
  const [stencils, setStencils] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [message, setMessage] = useState(null);

  useEffect(() => {
    loadStencils();
  }, [filter]);

  const loadStencils = async () => {
    setLoading(true);
    try {
      const params = filter !== 'all' ? { side: filter } : {};
      const result = await apiClient.listStencils(params);
      setStencils(result.stencils || []);
    } catch (err) {
      console.error('Failed to load stencils:', err);
      setMessage({ type: 'error', text: 'Failed to load stencils' });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this stencil?')) {
      return;
    }

    try {
      await apiClient.deleteStencil(id);
      setMessage({ type: 'success', text: 'Stencil deleted successfully' });
      loadStencils();

      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      console.error('Delete error:', err);
      setMessage({ type: 'error', text: 'Failed to delete stencil' });
    }
  };

  const handleExport = async (id, format) => {
    try {
      const result = await apiClient.exportStencil(id, format);

      // Download file
      const blob = new Blob(
        [format === 'svg' ? result.content : JSON.stringify(JSON.parse(result.content), null, 2)],
        { type: format === 'svg' ? 'image/svg+xml' : 'application/json' }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stencil_${id.substring(0, 8)}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setMessage({ type: 'success', text: `Exported as ${format.toUpperCase()}` });
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      console.error('Export error:', err);
      setMessage({ type: 'error', text: 'Failed to export stencil' });
    }
  };

  return (
    <div className="library-page">
      <div className="library-header">
        <h2>Stencil Library</h2>
        <button
          onClick={() => navigate('/')}
          className="btn btn-primary"
        >
          + New Stencil
        </button>
      </div>

      {message && (
        <div className={`${message.type}-message`}>
          {message.text}
        </div>
      )}

      <div className="library-filters">
        <button
          onClick={() => setFilter('all')}
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
        >
          All
        </button>
        <button
          onClick={() => setFilter('left')}
          className={`filter-btn ${filter === 'left' ? 'active' : ''}`}
        >
          Left
        </button>
        <button
          onClick={() => setFilter('right')}
          className={`filter-btn ${filter === 'right' ? 'active' : ''}`}
        >
          Right
        </button>
      </div>

      {loading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading stencils...</p>
        </div>
      ) : stencils.length === 0 ? (
        <div className="empty-state card">
          <div className="empty-icon">📂</div>
          <h3>No Stencils Yet</h3>
          <p>Upload a photo to create your first brow stencil</p>
          <button
            onClick={() => navigate('/')}
            className="btn btn-primary"
          >
            Get Started
          </button>
        </div>
      ) : (
        <div className="stencils-grid">
          {stencils.map(stencil => (
            <StencilCard
              key={stencil.stencil_id}
              stencil={stencil}
              onDelete={handleDelete}
              onExport={handleExport}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default LibraryPage;
