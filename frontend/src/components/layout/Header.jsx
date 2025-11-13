import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Header.css';

const Header = () => {
  const location = useLocation();

  return (
    <header className="header">
      <div className="header-container">
        <h1 className="header-title">Brow Stencil App</h1>
        <nav className="header-nav">
          <Link
            to="/"
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Upload
          </Link>
          <Link
            to="/library"
            className={`nav-link ${location.pathname === '/library' ? 'active' : ''}`}
          >
            Library
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
