import React, { useState, useEffect } from 'react'
import axios from 'axios'

const UploadedFiles = ({ onFileSelect, selectedFileId }) => {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchUploadedFiles()
  }, [])

  const fetchUploadedFiles = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/files/')
      setFiles(response.data)
      setError(null)
    } catch (error) {
      setError('Failed to fetch uploaded files')
      console.error('Error fetching files:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteFile = async (fileId) => {
    if (!window.confirm('Are you sure you want to delete this file and all its data?')) {
      return
    }

    try {
      await axios.delete(`/api/files/${fileId}/`)
      // Remove the file from state
      setFiles(files.filter(file => file.id !== fileId))
      
      // If this was the selected file, clear selection
      if (selectedFileId === fileId) {
        onFileSelect?.(null)
      }
    } catch (error) {
      console.error('Error deleting file:', error)
      alert('Failed to delete file. Please try again.')
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString()
  }

  if (loading) {
    return (
      <div className="uploaded-files-container">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading uploaded files...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="uploaded-files-container">
        <div className="error">
          <p>{error}</p>
          <button onClick={fetchUploadedFiles} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="uploaded-files-container">
      <div className="files-header">
        <h3>📁 Uploaded Files ({files.length})</h3>
        <button onClick={fetchUploadedFiles} className="refresh-button">
          🔄 Refresh
        </button>
      </div>

      {files.length === 0 ? (
        <div className="no-files">
          <p>No files uploaded yet. Upload an Excel file to get started!</p>
        </div>
      ) : (
        <div className="files-list">
          {files.map(file => (
            <div 
              key={file.id} 
              className={`file-item ${selectedFileId === file.id ? 'selected' : ''}`}
              onClick={() => onFileSelect?.(file)}
            >
              <div className="file-header">
                <div className="file-name">
                  <span className="file-icon">
                    {file.original_filename.endsWith('.csv') ? '📄' : '📊'}
                  </span>
                  <strong>{file.original_filename}</strong>
                </div>
                <div className="file-actions">
                  <button 
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteFile(file.id)
                    }}
                    className="delete-button"
                    title="Delete file"
                  >
                    🗑️
                  </button>
                </div>
              </div>

              <div className="file-details">
                <div className="file-stats">
                  <span className="stat">
                    📊 {file.sheet_count} sheet{file.sheet_count !== 1 ? 's' : ''}
                  </span>
                  <span className="stat">
                    📈 {file.row_count.toLocaleString()} rows
                  </span>
                  <span className="stat">
                    📋 {file.column_count} columns
                  </span>
                  <span className="stat">
                    💾 {formatFileSize(file.file_size)}
                  </span>
                </div>

                <div className="file-meta">
                  <span className="upload-date">
                    📅 Uploaded: {formatDate(file.uploaded_at)}
                  </span>
                  <span className={`status ${file.processed ? 'processed' : 'processing'}`}>
                    {file.processed ? '✅ Processed' : '⏳ Processing...'}
                  </span>
                </div>

                {file.processing_errors && (
                  <div className="processing-errors">
                    <span className="error-icon">⚠️</span>
                    <span>{file.processing_errors}</span>
                  </div>
                )}

                {file.sheets && file.sheets.length > 0 && (
                  <div className="sheets-info">
                    <strong>Tables created:</strong>
                    {file.sheets.map((sheet, index) => (
                      <span key={index} className="sheet-tag">
                        {sheet.sheet_name} ({sheet.row_count} rows)
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default UploadedFiles