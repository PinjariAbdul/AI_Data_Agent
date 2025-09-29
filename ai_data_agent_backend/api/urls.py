from django.urls import path
from . import views
# Temporarily comment out enhanced views to fix imports
# from .enhanced_views import (
#     advanced_analytical_query,
#     business_intelligence_analysis,
#     analytical_capabilities,
#     generate_strategic_insights
# )

urlpatterns = [
    path('query/', views.QueryProcessorView.as_view(), name='query_processor'),
    path('schema/', views.SchemaAnalysisView.as_view(), name='schema_analysis'),
    path('health/', views.database_health, name='database_health'),
    path('populate/', views.populate_sample_data, name='populate_sample_data'),
    path('sql/', views.DirectSQLView.as_view(), name='direct_sql'),
    
    # File upload endpoints
    path('upload/', views.FileUploadView.as_view(), name='file_upload'),
    path('files/', views.UploadedFilesView.as_view(), name='uploaded_files'),
    path('files/<int:file_id>/', views.UploadedFilesView.as_view(), name='delete_uploaded_file'),
    
    # Enhanced analytical endpoints - temporarily disabled for testing
    # path('advanced-query/', advanced_analytical_query, name='advanced_analytical_query'),
    # path('business-intelligence/', business_intelligence_analysis, name='business_intelligence_analysis'),
    # path('capabilities/', analytical_capabilities, name='analytical_capabilities'),
    # path('strategic-insights/', generate_strategic_insights, name='generate_strategic_insights'),
]