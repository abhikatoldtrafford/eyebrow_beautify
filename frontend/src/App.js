import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/layout/Header';
import UploadPage from './components/upload/UploadPage';
import EditorPage from './components/editor/EditorPage';
import LibraryPage from './components/library/LibraryPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Header />
        <main className="container">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/editor" element={<EditorPage />} />
            <Route path="/library" element={<LibraryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
