import React, { useState } from 'react'
import axios from 'axios'
import QueryResponse from './QueryResponse'

const ChatInterface = ({ selectedFile }) => {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const result = await axios.post('/api/query/', {
        question: question.trim(),
        query_context: selectedFile ? `Analyzing data from file: ${selectedFile.original_filename}` : ''
      })
      
      setResponse(result.data)
    } catch (err) {
      console.error('Query error:', err)
      setError(
        err.response?.data?.error || 
        'Failed to process question. Please check if the backend server is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  const getExampleQuestions = () => {
    if (selectedFile) {
      // File-specific example questions
      return [
        "How many rows of data do we have?",
        "What are the column names in this dataset?",
        "Show me a summary of the data",
        "Are there any missing values?",
        "What are the data types of each column?",
        "Show me the first few records"
      ]
    } else {
      // Default example questions for pre-existing data
      return [
        "How many customers do we have?",
        "What are our top-selling products?",
        "Show me sales trends by region",
        "Which customers haven't logged in recently?",
        "What's our average order value?",
        "Show me employee distribution by department"
      ]
    }
  }

  const exampleQuestions = getExampleQuestions()

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>💬 Ask Your Business Questions</h2>
        <p>
          {selectedFile 
            ? `Ask questions about ${selectedFile.original_filename}` 
            : "Ask complex questions about your data in natural language"
          }
        </p>
        {!selectedFile && (
          <div className="no-file-warning">
            ⚠️ No file selected. Upload and select an Excel file to analyze your own data, or use the default sample data.
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="chat-input-area">
        <textarea
          className="chat-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={
            selectedFile 
              ? `e.g., What trends can you see in the ${selectedFile.original_filename} data?`
              : "e.g., What are our top-selling products this quarter?"
          }
          rows={3}
          disabled={loading}
        />
        <button 
          type="submit" 
          className="submit-button"
          disabled={loading || !question.trim()}
        >
          {loading ? 'Processing...' : 'Ask Question'}
        </button>
      </form>

      <div className="example-questions">
        <h3>Example Questions:</h3>
        <div className="example-grid">
          {exampleQuestions.map((example, index) => (
            <button
              key={index}
              className="example-button"
              onClick={() => setQuestion(example)}
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Processing your question...</span>
        </div>
      )}

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {response && <QueryResponse response={response} />}
    </div>
  )
}

export default ChatInterface