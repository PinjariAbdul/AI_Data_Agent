import React, { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import DatabaseHealth from './components/DatabaseHealth'
import SchemaViewer from './components/SchemaViewer'
import FileUpload from './components/FileUpload'
import UploadedFiles from './components/UploadedFiles'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadMessage, setUploadMessage] = useState(null)

  const handleUploadSuccess = (result) => {
    setUploadMessage({
      type: 'success',
      text: `File uploaded successfully! Created ${result.tables_created.length} table(s).`
    })
    // Auto-switch to files tab to see the uploaded file
    setActiveTab('files')
    // Clear message after 5 seconds
    setTimeout(() => setUploadMessage(null), 5000)
  }

  const handleUploadError = (error) => {
    setUploadMessage({
      type: 'error',
      text: `Upload failed: ${error}`
    })
    // Clear message after 10 seconds
    setTimeout(() => setUploadMessage(null), 10000)
  }

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    if (file) {
      // Auto-switch to chat tab when a file is selected
      setActiveTab('chat')
    }
  }

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤖 AI Data Agent</h1>
        <p>Upload Excel files and ask complex business questions in natural language</p>
        
        {uploadMessage && (
          <div className={`upload-message ${uploadMessage.type}`}>
            {uploadMessage.text}
            <button 
              onClick={() => setUploadMessage(null)}
              className="close-message"
            >
              ×
            </button>
          </div>
        )}
        
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📤 Upload
          </button>
          <button 
            className={`nav-tab ${activeTab === 'files' ? 'active' : ''}`}
            onClick={() => setActiveTab('files')}
          >
            📁 Files
          </button>
          <button 
            className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat
          </button>
          <button 
            className={`nav-tab ${activeTab === 'health' ? 'active' : ''}`}
            onClick={() => setActiveTab('health')}
          >
            📊 Database Health
          </button>
          <button 
            className={`nav-tab ${activeTab === 'schema' ? 'active' : ''}`}
            onClick={() => setActiveTab('schema')}
          >
            🗄️ Schema
          </button>
        </nav>
      </header>

      <main className="app-main">
        {activeTab === 'upload' && (
          <div className="upload-section">
            <FileUpload 
              onUploadSuccess={handleUploadSuccess}
              onUploadError={handleUploadError}
            />
          </div>
        )}
        {activeTab === 'files' && (
          <UploadedFiles 
            onFileSelect={handleFileSelect}
            selectedFileId={selectedFile?.id}
          />
        )}
        {activeTab === 'chat' && (
          <div className="chat-section">
            {selectedFile && (
              <div className="selected-file-info">
                <h3>📊 Analyzing: {selectedFile.original_filename}</h3>
                <p>Ask questions about your data using natural language!</p>
              </div>
            )}
            <ChatInterface selectedFile={selectedFile} />
          </div>
        )}
        {activeTab === 'health' && <DatabaseHealth />}
        {activeTab === 'schema' && <SchemaViewer />}
      </main>

      <footer className="app-footer">
        <p>Built with React + Django REST Framework | AI-powered analysis of Excel files</p>
        <p>🚀 Upload any Excel file format • Handle dirty data • Ask complex questions</p>
      </footer>
    </div>
  )
}

export default App