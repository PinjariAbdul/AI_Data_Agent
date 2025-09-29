import React, { useState, useEffect } from 'react'
import axios from 'axios'

const DatabaseHealth = () => {
  const [healthData, setHealthData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchHealthData()
  }, [])

  const fetchHealthData = async () => {
    try {
      setLoading(true)
      const response = await axios.get('/api/health/')
      setHealthData(response.data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch database health information')
      console.error('Health check error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ok': return '✅'
      case 'empty': return '⚠️'
      case 'error': return '❌'
      default: return '❓'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'ok': return '#4CAF50'
      case 'empty': return '#FF9800'
      case 'error': return '#F44336'
      default: return '#9E9E9E'
    }
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        <span>Checking database health...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-message">
        <strong>Error:</strong> {error}
        <button onClick={fetchHealthData} className="retry-button">
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="health-container">
      <div className="health-header">
        <h2>📊 Database Health Status</h2>
        <p>Overview of database tables and record counts</p>
        <button onClick={fetchHealthData} className="refresh-button">
          🔄 Refresh
        </button>
      </div>

      {healthData && (
        <div className="health-content">
          <div className="health-summary">
            <h3>Database Summary</h3>
            <p>Last checked: {new Date(healthData.timestamp).toLocaleString()}</p>
          </div>

          <div className="tables-grid">
            {Object.entries(healthData.database_health).map(([tableName, tableInfo]) => (
              <div key={tableName} className="table-card">
                <div className="table-header">
                  <span className="table-icon">
                    {getStatusIcon(tableInfo.status)}
                  </span>
                  <h4>{tableName}</h4>
                </div>
                
                <div className="table-info">
                  <div className="record-count">
                    <strong>{tableInfo.count.toLocaleString()}</strong>
                    <span>records</span>
                  </div>
                  
                  <div 
                    className="status-indicator"
                    style={{ backgroundColor: getStatusColor(tableInfo.status) }}
                  >
                    {tableInfo.status.toUpperCase()}
                  </div>
                  
                  {tableInfo.error && (
                    <div className="table-error">
                      Error: {tableInfo.error}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="health-stats">
            <h3>Statistics</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <strong>Total Tables:</strong>
                <span>{Object.keys(healthData.database_health).length}</span>
              </div>
              <div className="stat-item">
                <strong>Active Tables:</strong>
                <span>
                  {Object.values(healthData.database_health).filter(t => t.status === 'ok').length}
                </span>
              </div>
              <div className="stat-item">
                <strong>Total Records:</strong>
                <span>
                  {Object.values(healthData.database_health)
                    .reduce((sum, table) => sum + table.count, 0)
                    .toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DatabaseHealth