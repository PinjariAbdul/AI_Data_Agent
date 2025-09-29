import React, { useState, useCallback } from 'react'
import axios from 'axios'

const FileUpload = ({ onUploadSuccess, onUploadError }) => {
  const [uploading, setUploading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file) => {
    // Validate file type
    const allowedTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel', // .xls
      'text/csv' // .csv
    ]
    
    const fileExtension = file.name.split('.').pop().toLowerCase()
    const allowedExtensions = ['xlsx', 'xls', 'csv']
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      onUploadError?.('Please upload a valid Excel file (.xlsx, .xls) or CSV file (.csv)')
      return
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      onUploadError?.('File size must be less than 50MB')
      return
    }

    setUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(
        '/api/upload/', 
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            )
            setUploadProgress(progress)
          }
        }
      )

      if (response.data.success) {
        onUploadSuccess?.(response.data)
      } else {
        onUploadError?.(response.data.error || 'Upload failed')
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || 
                          error.message || 
                          'An error occurred during upload'
      onUploadError?.(errorMessage)
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }

  return (
    <div className="file-upload-container">
      <div 
        className={`file-upload-dropzone ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className="upload-progress">
            <div className="progress-spinner"></div>
            <p>Uploading and processing your file...</p>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <span className="progress-text">{uploadProgress}%</span>
          </div>
        ) : (
          <>
            <div className="upload-icon">📊</div>
            <h3>Upload Your Excel File</h3>
            <p>Drag and drop your Excel file here, or click to browse</p>
            <p className="upload-note">
              Supports .xlsx, .xls, and .csv files (max 50MB)
            </p>
            <p className="upload-features">
              ✅ Any Excel file format<br/>
              ✅ Handles bad/inconsistent data<br/>
              ✅ Works with unnamed columns<br/>
              ✅ Processes dirty data automatically
            </p>
            <input
              type="file"
              id="file-input"
              className="file-input"
              accept=".xlsx,.xls,.csv"
              onChange={handleFileInput}
              disabled={uploading}
            />
            <label htmlFor="file-input" className="file-input-label">
              Choose File
            </label>
          </>
        )}
      </div>
    </div>
  )
}

export default FileUpload