import React, { useState, useEffect } from 'react'
import axios from 'axios'

const SchemaViewer = () => {
  const [schemaData, setSchemaData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedTable, setSelectedTable] = useState(null)

  useEffect(() => {
    fetchSchemaData()
  }, [])

  const fetchSchemaData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/schema/')
      console.log('Schema data received:', response.data)
      setSchemaData(response.data)
      setError(null)
    } catch (err) {
      console.error('Schema fetch error:', err)
      setError('Failed to fetch schema information')
    } finally {
      setLoading(false)
    }
  }

  const renderDataQualityIssues = (issues) => {
    if (!issues || issues.length === 0) {
      return <span className="no-issues">✅ No major issues detected</span>
    }

    return (
      <div className="quality-issues">
        {issues.map((issue, index) => (
          <div key={index} className="issue-item">
            <span className="issue-icon">⚠️</span>
            <div className="issue-details">
              <strong>{issue.type.replace('_', ' ').toUpperCase()}:</strong>
              {issue.column && <span> Column: {issue.column}</span>}
              {issue.percentage && <span> ({issue.percentage}%)</span>}
              {issue.count && <span> ({issue.count} records)</span>}
              {issue.message && <span> {issue.message}</span>}
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderColumns = (columns) => {
    return (
      <div className="columns-list">
        {columns.map((column, index) => (
          <div key={index} className="column-item">
            <div className="column-header">
              <strong>{column.name}</strong>
              <span className="column-type">{column.type}</span>
            </div>
            <div className="column-details">
              {column.primary_key && <span className="column-badge primary">PRIMARY KEY</span>}
              {column.nullable && <span className="column-badge nullable">NULLABLE</span>}
              {column.default && <span className="column-badge default">DEFAULT: {column.default}</span>}
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderSampleData = (sampleData) => {
    if (!sampleData || sampleData.length === 0) {
      return <p>No sample data available</p>
    }

    return (
      <div className="sample-data">
        <table className="data-table small">
          <tbody>
            {sampleData.slice(0, 3).map((row, index) => (
              <tr key={index}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    {cell !== null && cell !== undefined ? String(cell) : 'NULL'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        <span>Loading schema information...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-message">
        <strong>Error:</strong> {error}
        <button onClick={fetchSchemaData} className="retry-button">
          Retry
        </button>
      </div>
    )
  }

  console.log('Rendering schema data:', schemaData)

  return (
    <div className="schema-container">
      <div className="schema-header">
        <h2>🗄️ Database Schema Analysis</h2>
        <p>Explore tables, columns, and data quality issues</p>
      </div>

      {schemaData && schemaData.length > 0 ? (
        <div className="schema-content">
          <div className="tables-overview">
            <h3>Database Tables ({schemaData.length})</h3>
            <div className="tables-grid">
              {schemaData.map((table, index) => (
                <div 
                  key={index} 
                  className={`table-overview-card ${selectedTable === index ? 'selected' : ''}`}
                  onClick={() => setSelectedTable(selectedTable === index ? null : index)}
                >
                  <div className="table-overview-header">
                    <h4>{table.table_name}</h4>
                    <div className="table-stats">
                      <span>{table.columns.length} columns</span>
                      <span>{table.total_rows} rows</span>
                    </div>
                  </div>
                  
                  <div className="table-overview-quality">
                    {table.data_quality_issues && table.data_quality_issues.length > 0 ? (
                      <span className="quality-warning">
                        ⚠️ {table.data_quality_issues.length} quality issues
                      </span>
                    ) : (
                      <span className="quality-good">✅ Good quality</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedTable !== null && (
            <div className="table-details">
              <div className="table-details-header">
                <h3>📋 {schemaData[selectedTable].table_name} Details</h3>
                <button 
                  className="close-details"
                  onClick={() => setSelectedTable(null)}
                >
                  ✕
                </button>
              </div>

              <div className="details-sections">
                <div className="details-section">
                  <h4>Columns ({schemaData[selectedTable].columns.length})</h4>
                  {renderColumns(schemaData[selectedTable].columns)}
                </div>

                <div className="details-section">
                  <h4>Data Quality Issues</h4>
                  {renderDataQualityIssues(schemaData[selectedTable].data_quality_issues)}
                </div>

                <div className="details-section">
                  <h4>Sample Data</h4>
                  {renderSampleData(schemaData[selectedTable].sample_data)}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="no-data">
          <p>No schema data available. Please upload some files first.</p>
        </div>
      )}
    </div>
  )
}

export default SchemaViewer