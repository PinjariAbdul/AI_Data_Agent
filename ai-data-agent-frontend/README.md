# AI Data Agent Frontend

React frontend for the AI Data Agent - a conversational interface for complex business questions from SQL databases.

## 🚀 Features

- **Conversational Interface**: Natural language input for business questions
- **Real-time Results**: Instant SQL generation and data visualization
- **Database Health Monitoring**: Visual overview of database status
- **Schema Explorer**: Interactive database schema analysis
- **Responsive Design**: Works on desktop and mobile devices
- **Data Visualization**: Charts and tables for query results

## 🛠️ Technology Stack

- **React 18** with hooks and functional components
- **Vite** for fast development and building
- **Axios** for API communication
- **CSS3** with modern grid and flexbox layouts
- **Responsive Design** with mobile-first approach

## 📋 Prerequisites

- Node.js 16+ and npm
- Backend server running on http://127.0.0.1:8002
- Modern web browser

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
cd ai-data-agent-frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### 3. Build for Production
```bash
npm run build
```

## 🎯 Usage

### Chat Interface
1. Type natural language questions in the input area
2. Click "Ask Question" or press Enter
3. View generated SQL, results, and natural language explanations
4. Explore data visualizations when available

### Example Questions
- "How many customers do we have?"
- "What are our top-selling products?"
- "Show me sales trends by region"
- "Which customers haven't logged in recently?"
- "What's our average order value?"

### Database Health
- View all database tables and record counts
- Monitor table status (OK, Empty, Error)
- Check data quality indicators
- Refresh health information

### Schema Explorer
- Browse all database tables
- Explore column definitions and data types
- Identify data quality issues
- View sample data from each table

## 🔌 API Integration

The frontend communicates with the Django backend through these endpoints:

- `POST /api/query/` - Submit natural language questions
- `GET /api/health/` - Get database health status
- `GET /api/schema/` - Retrieve schema information

API requests are proxied through Vite's development server to avoid CORS issues.

## 🎨 Components Structure

```
src/
├── components/
│   ├── ChatInterface.jsx      # Main chat interface
│   ├── QueryResponse.jsx      # Display query results
│   ├── DatabaseHealth.jsx     # Database status view
│   └── SchemaViewer.jsx       # Schema exploration
├── App.jsx                    # Main application component
├── main.jsx                   # Application entry point
├── index.css                  # Global styles
└── App.css                    # App-specific styles
```

## 🎯 Key Features

### Intelligent Query Processing
- Real-time natural language processing
- SQL query generation with confidence scoring
- Contextual business explanations
- Error handling and user feedback

### Data Visualization
- Automatic chart generation based on data types
- Interactive tables with pagination
- Responsive design for various screen sizes
- Export capabilities for results

### User Experience
- Loading states and progress indicators
- Error messages with retry options
- Example questions to guide users
- Intuitive navigation between features

## 🔧 Configuration

### Environment Variables
Create a `.env` file if needed:
```env
VITE_API_BASE_URL=http://127.0.0.1:8002
```

### Proxy Configuration
The Vite configuration includes API proxying to the backend:
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8002',
      changeOrigin: true
    }
  }
}
```

## 🚀 Production Deployment

1. Build the application:
   ```bash
   npm run build
   ```

2. Serve the `dist` folder using any static file server:
   ```bash
   npm run preview
   ```

3. For production, consider using:
   - Nginx for static file serving
   - CDN for global distribution
   - Environment-specific API endpoints

## 🐛 Troubleshooting

### Common Issues

**API Connection Failed**
- Ensure Django backend is running on port 8000
- Check CORS configuration in Django settings
- Verify network connectivity

**Build Errors**
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version compatibility
- Verify all dependencies are installed

**Styling Issues**
- Check browser developer tools for CSS conflicts
- Ensure all CSS files are properly imported
- Verify responsive design in different viewports

## 📱 Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🔄 Development Workflow

1. Start backend server: `python manage.py runserver`
2. Start frontend development: `npm run dev`
3. Make changes and see live updates
4. Test API integration with backend
5. Build and deploy for production

## 📝 Notes

- The frontend is designed to work with the Django backend's specific API format
- All API responses are expected to follow the backend's serializer structure
- Error handling includes both network errors and application-level errors
- The interface adapts to different data types and visualization needs