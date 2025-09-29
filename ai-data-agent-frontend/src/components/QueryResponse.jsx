import React from 'react'
import Plot from 'react-plotly.js'

const QueryResponse = ({ response }) => {
  const renderTable = (data) => {
    if (!data || data.length === 0) return <p>No data returned</p>

    const columns = Object.keys(data[0])
    
    return (
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map(col => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 10).map((row, index) => (
              <tr key={index}>
                {columns.map(col => (
                  <td key={col}>
                    {row[col] !== null && row[col] !== undefined 
                      ? String(row[col]) 
                      : 'N/A'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 10 && (
          <p className="table-note">
            Showing first 10 of {data.length} results
          </p>
        )}
      </div>
    )
  }

  const renderCharts = (charts) => {
    console.log('Rendering charts:', charts); // Debug log
    
    if (!charts || Object.keys(charts).length === 0) {
      console.log('No charts to render');
      return null;
    }

    return (
      <div className="charts-section">
        <h3>📊 Visualizations</h3>
        {Object.entries(charts).map(([chartType, chartData]) => {
          console.log(`Rendering chart type: ${chartType}`, chartData); // Debug log
          
          if (chartType === 'error') {
            return (
              <div key={chartType} className="chart-error">
                <p>Visualization Error: {chartData}</p>
              </div>
            )
          }
          
          if (chartType === 'table') {
            return null // Skip table here as we render it separately
          }

          // Handle the new visualization structure from backend
          if (chartType.includes('chart') && chartData && typeof chartData === 'string' && chartData.includes('plotly')) {
            // Try to extract data from Plotly HTML
            try {
              // This is a more robust approach to handle Plotly charts
              const divMatch = chartData.match(/Plotly\.newPlot\("([^"]+)",\s*(\[[^\]]*\])\s*,\s*(\{[^}]*\})/);
              if (divMatch) {
                const chartId = divMatch[1];
                const chartDataArray = JSON.parse(divMatch[2]);
                const chartLayout = JSON.parse(divMatch[3]);
                
                // Ensure layout has proper sizing
                const layout = {
                  ...chartLayout,
                  autosize: true,
                  height: 400,
                  margin: {
                    l: 50,
                    r: 50,
                    t: 50,
                    b: 50,
                    ...chartLayout.margin
                  }
                };
                
                return (
                  <div key={chartType} className="chart-container">
                    <h4>{chartType.replace('_', ' ').replace('chart', 'Chart').toUpperCase()}</h4>
                    <Plot
                      data={chartDataArray}
                      layout={layout}
                      config={{ responsive: true }}
                      style={{ width: '100%', height: '100%' }}
                      useResizeHandler={true}
                    />
                  </div>
                );
              }
            } catch (e) {
              console.error('Error parsing Plotly chart:', e);
              // Fallback to rendering HTML if parsing fails
            }
          }

          // Handle HTML string visualizations
          if (typeof chartData === 'string' && chartData.trim().startsWith('<')) {
            return (
              <div key={chartType} className="chart-container">
                <h4>{chartType.replace('_', ' ').replace('chart', 'Chart').toUpperCase()}</h4>
                <div 
                  dangerouslySetInnerHTML={{ __html: chartData }}
                  style={{ background: 'white', padding: '10px', borderRadius: '8px', border: '1px solid #ddd', minHeight: '400px' }}
                />
              </div>
            )
          }

          // Handle error messages in chart data
          if (typeof chartData === 'string' && chartData.includes('failed')) {
            return (
              <div key={chartType} className="chart-error">
                <p>Chart Generation Error: {chartData}</p>
              </div>
            )
          }

          // Fallback for any other chart data
          return (
            <div key={chartType} className="chart-container">
              <h4>{chartType.replace('_', ' ').replace('chart', 'Chart').toUpperCase()}</h4>
              <div>
                {typeof chartData === 'object' ? (
                  <pre>{JSON.stringify(chartData, null, 2)}</pre>
                ) : (
                  <p>{String(chartData)}</p>
                )}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  console.log('Query response:', response); // Debug log

  // Map backend response fields to frontend expected fields
  const mappedResponse = {
    ...response,
    result_data: response.results || response.result_data,
    natural_language_answer: response.answer || response.natural_language_answer,
    confidence_score: response.confidence || response.confidence_score,
    query_explanation: response.analysis || response.query_explanation,
    charts: response.visualizations || response.charts
  };

  return (
    <div className="response-container">
      <div className="response-header">
        <h3>📝 Query Results</h3>
        <div className="confidence-score">
          Confidence: {(mappedResponse.confidence_score * 100).toFixed(1)}%
        </div>
      </div>

      <div className="response-section">
        <h4>🔍 Generated SQL Query:</h4>
        <pre className="sql-query">{mappedResponse.sql_query}</pre>
      </div>

      <div className="response-section">
        <h4>💬 Natural Language Answer:</h4>
        <div className="natural-answer">
          {mappedResponse.natural_language_answer}
        </div>
      </div>

      {mappedResponse.result_data && mappedResponse.result_data.length > 0 && (
        <div className="response-section">
          <h4>📊 Data Results:</h4>
          {renderTable(mappedResponse.result_data)}
        </div>
      )}

      {mappedResponse.charts && (
        <div className="response-section">
          {renderCharts(mappedResponse.charts)}
        </div>
      )}

      <div className="response-section">
        <h4>🔍 Query Analysis:</h4>
        <pre className="query-explanation">{JSON.stringify(mappedResponse.query_explanation, null, 2)}</pre>
      </div>
    </div>
  )
}

export default QueryResponse