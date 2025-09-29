# AI Data Agent Backend

A sophisticated conversational interface that answers complex business questions from SQL databases with poor schemas and dirty data.

## 🚀 Features

- **AI-Powered Query Generation**: Convert natural language questions to SQL queries
- **Bad Schema Handling**: Works with poorly named columns and tables
- **Data Quality Analysis**: Handles missing values, duplicates, and inconsistent data
- **Visualization Support**: Generates charts and tables for query results
- **Natural Language Responses**: Explains results in business-friendly language

## 🛠️ Technology Stack

**Backend:**
- Django + Django REST Framework
- Database: SQLite (can be extended to PostgreSQL)
- AI: OpenAI GPT-4 for query generation and analysis
- Data Processing: Pandas, Matplotlib, Plotly, Seaborn
- API: RESTful API with comprehensive endpoints

**Frontend:**
- React 18 with modern hooks
- Vite for fast development
- Axios for API communication
- Responsive CSS with Grid/Flexbox
- Real-time data visualization

## 📋 Requirements

**Backend:**
- Python 3.8+
- OpenAI API Key (for AI features)
- Virtual environment (recommended)

**Frontend:**
- Node.js 16+
- npm or yarn
- Modern web browser

## 🔧 Setup Instructions

### ⚠️ IMPORTANT: OpenAI API Key Configuration

**Before starting the application, you MUST configure your OpenAI API key:**

1. **Get your API key** from [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. **Update the `.env` file** in `ai_data_agent_backend/.env`:
   ```env
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```
3. **Restart the Django server** after updating the key

**Without a valid API key, the AI features will not work and you'll see authentication errors.**

### Backend Setup

#### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv env1
env1\Scripts\activate  # Windows
# source env1/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### 2. Database Setup
```bash
cd ai_data_agent_backend
python manage.py makemigrations
python manage.py migrate
python manage.py populate_sample_data
```

#### 3. Configuration
Create `.env` file in `ai_data_agent_backend/` directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### 4. Start Backend Server
```bash
python manage.py runserver
```
Backend will be available at `http://127.0.0.1:8002`

### Frontend Setup

#### 1. Install Dependencies
```bash
cd ai-data-agent-frontend
npm install
```

#### 2. Start Frontend Development Server
```bash
npm run dev
```
Frontend will be available at `http://localhost:3000`

#### 3. Build for Production
```bash
npm run build
```

## 📡 API Endpoints

### Core Endpoints

#### 1. Process Natural Language Query
```http
POST /api/query/
Content-Type: application/json

{
    "question": "What are our top-selling products this quarter?",
    "query_context": "Focus on revenue analysis"
}
```

**Response:**
```json
{
    "question": "What are our top-selling products this quarter?",
    "sql_query": "SELECT col1, SUM(c3) as revenue FROM prod_tbl p JOIN...",
    "result_data": [...],
    "natural_language_answer": "Based on the analysis...",
    "charts": {...},
    "confidence_score": 0.85,
    "query_explanation": "Query Analysis: Product revenue analysis..."
}
```

#### 2. Database Schema Analysis
```http
GET /api/schema/
```

**Response:**
```json
[
    {
        "table_name": "cust_data",
        "columns": [...],
        "total_rows": 5,
        "data_quality_issues": [...]
    }
]
```

#### 3. Direct SQL Execution
```http
POST /api/sql/
Content-Type: application/json

{
    "sql_query": "SELECT * FROM cust_data LIMIT 5"
}
```

#### 4. Database Health Check
```http
GET /api/health/
```

## 🗄️ Database Schema

The system works with intentionally complex/poor database schemas to demonstrate AI capabilities:

### Tables Overview
- **cust_data**: Customer information (poorly named columns)
- **prod_tbl**: Product catalog (columns named col1, col2, etc.)
- **Order_History**: Order records (mixed naming conventions)
- **orderitems**: Order line items (inconsistent casing)
- **sales_data_2023**: Sales data (cryptic column names a1, b2, c3...)
- **EMP_RECORDS**: Employee information

### Data Quality Issues
- Missing values in many fields
- Inconsistent date formats
- Mixed case in text fields
- Duplicate records
- Orphaned foreign key relationships

## 🧪 Testing

### Backend API Testing
```bash
# Run comprehensive API test
python api_tester.py

# Test specific endpoints
python api_tester.py health
python api_tester.py schema
python api_tester.py sql "SELECT COUNT(*) FROM cust_data"
python api_tester.py ai "Show me customer distribution by city"
```

### Frontend Testing
1. Open http://localhost:3000 in your browser
2. Try the example questions in the chat interface
3. Explore the Database Health and Schema tabs
4. Test responsive design on different screen sizes

### Example Questions to Try
- "How many customers do we have?"
- "What are our top-selling products?"
- "Show me sales trends by region"
- "Which customers haven't logged in recently?"
- "What's our average order value?"
- "Show me employee distribution by department"

## 🎯 Key Challenges Solved

1. **Bad Schema Navigation**: AI understands poorly named columns
2. **Data Quality Issues**: Handles missing/inconsistent data gracefully
3. **Complex Queries**: Generates sophisticated JOINs and aggregations
4. **Business Context**: Translates technical results to business insights
5. **Visualization**: Creates appropriate charts for different data types

## 🎯 Project Structure

```
AI Data Agent/
├── ai_data_agent_backend/     # Django Backend
│   ├── ai_data_agent_backend/ # Django settings
│   ├── api/                   # Main API app
│   │   ├── models.py         # Database models
│   │   ├── views.py          # API endpoints
│   │   ├── ai_agent.py       # AI query processor
│   │   └── serializers.py    # API serializers
│   ├── manage.py             # Django management
│   └── .env                  # Environment variables
├── ai-data-agent-frontend/    # React Frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── App.jsx          # Main app component
│   │   └── index.css        # Global styles
│   ├── package.json         # Frontend dependencies
│   └── vite.config.js       # Vite configuration
├── api_tester.py             # Backend testing utility
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🛡️ Error Handling

The system includes comprehensive error handling for:
- Malformed SQL queries
- Data quality issues
- AI service connectivity
- Missing data scenarios
- Schema inconsistencies

## 🚀 Production Considerations

For production deployment:
1. Use PostgreSQL instead of SQLite
2. Add authentication and authorization
3. Implement rate limiting
4. Add comprehensive logging
5. Use production WSGI server (Gunicorn)
6. Add environment-specific configurations

## 📝 Notes

- The system is designed to work with real-world messy data
- AI features require OpenAI API key
- Sample data includes intentional quality issues for testing
- Backend focuses on API functionality over frontend presentation