from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser, FormParser
from django.db import connection
from .serializers import *
from .ai_agent import SQLQueryAgent
from .excel_processor import ExcelProcessor
from .models import *
import json
import os
from django.http import JsonResponse
from datetime import datetime
from django.conf import settings

# Create your views here.

class QueryProcessorView(APIView):
    """
    Main API endpoint for processing natural language questions
    """
    
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        question = serializer.validated_data['question']
        context = serializer.validated_data.get('query_context', '')
        
        try:
            # Initialize AI agent
            agent = SQLQueryAgent()
            
            # Process the question
            result = agent.process_query(question, context)
            
            # Serialize the response
            response_serializer = QueryResponseSerializer(data=result)
            if response_serializer.is_valid():
                return Response(response_serializer.validated_data, status=status.HTTP_200_OK)
            else:
                return Response(response_serializer.errors, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except ValueError as ve:
            # Handle API key configuration errors
            return Response({
                'error': str(ve),
                'question': question,
                'setup_required': True
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            return Response({
                'error': f'Error processing question: {str(e)}',
                'question': question
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SchemaAnalysisView(APIView):
    """
    API endpoint for analyzing database schema and data quality
    """
    
    def get(self, request):
        try:
            # Get all tables and their column information
            with connection.cursor() as cursor:
                # Get table names
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' 
                    AND name NOT LIKE 'django_%' 
                    AND name NOT LIKE 'auth_%'
                    AND name NOT LIKE 'sqlite_%'
                """)
                
                tables = [row[0] for row in cursor.fetchall()]
                print(f"Found tables: {tables}")  # Debug log
                
                schema_analysis = []
                
                for table in tables:
                    try:
                        # Get column information
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = cursor.fetchall()
                        
                        # Get sample data
                        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                        sample_data = cursor.fetchall()
                        
                        # Analyze data quality
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        total_rows = cursor.fetchone()[0]
                        
                        table_analysis = {
                            'table_name': table,
                            'columns': [
                                {
                                    'name': col[1],
                                    'type': col[2],
                                    'nullable': not col[3],
                                    'default': col[4],
                                    'primary_key': bool(col[5])
                                } for col in columns
                            ],
                            'sample_data': sample_data,
                            'total_rows': total_rows,
                            'data_quality_issues': self._analyze_data_quality(table, cursor)
                        }
                        
                        schema_analysis.append(table_analysis)
                    except Exception as table_error:
                        print(f"Error processing table {table}: {str(table_error)}")
                        # Continue with other tables even if one fails
                        continue
                
                print(f"Returning schema analysis with {len(schema_analysis)} tables")
                return Response(schema_analysis, status=status.HTTP_200_OK)
                
        except Exception as e:
            print(f"Error in schema analysis: {str(e)}")  # Debug log
            return Response({
                'error': f'Error analyzing schema: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _analyze_data_quality(self, table_name, cursor):
        """
        Analyze data quality issues for a given table
        """
        issues = []
        
        try:
            # Check for NULL values
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            for column in columns:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column} IS NULL")
                    null_count = cursor.fetchone()[0]
                    
                    if null_count > 0:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        total_count = cursor.fetchone()[0]
                        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
                        
                        if null_percentage > 10:  # If more than 10% nulls
                            issues.append({
                                'type': 'high_null_percentage',
                                'column': column,
                                'percentage': round(null_percentage, 2)
                            })
                except Exception as column_error:
                    print(f"Error analyzing column {column} in table {table_name}: {str(column_error)}")
                    # Continue with other columns even if one fails
                    continue
            
            # Check for duplicate records
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT COUNT(DISTINCT *) FROM {table_name}")
                distinct_rows = cursor.fetchone()[0]
                
                if total_rows != distinct_rows:
                    duplicate_count = total_rows - distinct_rows
                    issues.append({
                        'type': 'duplicate_records',
                        'count': duplicate_count
                    })
            except Exception as duplicate_error:
                print(f"Error checking duplicates in table {table_name}: {str(duplicate_error)}")
                issues.append({
                    'type': 'analysis_error',
                    'message': f'Duplicate check failed: {str(duplicate_error)}'
                })
            
        except Exception as e:
            print(f"Error in data quality analysis for table {table_name}: {str(e)}")
            issues.append({
                'type': 'analysis_error',
                'message': str(e)
            })
        
        return issues

@api_view(['GET'])
def database_health(request):
    """
    Check overall database health and provide statistics
    """
    try:
        with connection.cursor() as cursor:
            health_info = {}
            
            # Get table counts
            tables = ['cust_data', 'prod_tbl', 'Order_History', 'orderitems', 'sales_data_2023', 'EMP_RECORDS']
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    health_info[table] = {'count': count, 'status': 'ok' if count > 0 else 'empty'}
                except Exception as e:
                    health_info[table] = {'count': 0, 'status': 'error', 'error': str(e)}
            
            return JsonResponse({
                'database_health': health_info,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return JsonResponse({
            'error': f'Error checking database health: {str(e)}'
        }, status=500)

@api_view(['POST'])
def populate_sample_data(request):
    """
    Populate the database with sample data for testing
    """
    try:
        # This would typically be done via Django management command
        # For now, return a message
        return JsonResponse({
            'message': 'Sample data population should be done via Django management command',
            'command': 'python manage.py populate_sample_data'
        })
    except Exception as e:
        return JsonResponse({
            'error': f'Error populating sample data: {str(e)}'
        }, status=500)

class DirectSQLView(APIView):
    """
    API endpoint for executing direct SQL queries (for testing/debugging)
    """
    
    def post(self, request):
        sql_query = request.data.get('sql_query', '')
        
        if not sql_query:
            return Response({
                'error': 'SQL query is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                
                if cursor.description:  # SELECT query
                    columns = [col[0] for col in cursor.description]
                    results = [
                        dict(zip(columns, row))
                        for row in cursor.fetchall()
                    ]
                    
                    return Response({
                        'results': results,
                        'row_count': len(results),
                        'columns': columns
                    }, status=status.HTTP_200_OK)
                else:  # INSERT/UPDATE/DELETE query
                    return Response({
                        'message': 'Query executed successfully',
                        'rows_affected': cursor.rowcount
                    }, status=status.HTTP_200_OK)
                    
        except Exception as e:
            return Response({
                'error': f'SQL execution error: {str(e)}',
                'sql_query': sql_query
            }, status=status.HTTP_400_BAD_REQUEST)

class FileUploadView(APIView):
    """
    API endpoint for uploading and processing Excel files
    """
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        uploaded_file = request.FILES.get('file')
        
        if not uploaded_file:
            return Response({
                'error': 'No file provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.xlsx', '.xls', '.csv']:
            return Response({
                'error': f'Unsupported file format: {file_extension}. Supported formats: .xlsx, .xls, .csv'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            return Response({
                'error': 'File size exceeds 50MB limit'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Create uploads directory if it doesn't exist
            upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # Generate unique filename
            processor = ExcelProcessor()
            safe_filename = processor._generate_safe_filename(uploaded_file.name)
            file_path = os.path.join(upload_dir, safe_filename)
            
            # Save file to disk
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # Process the file
            result = processor.process_uploaded_file(uploaded_file, file_path)
            
            if result['success']:
                return Response({
                    'success': True,
                    'message': 'File uploaded and processed successfully',
                    'file_id': result['file_id'],
                    'tables_created': result['tables_created'],
                    'metadata': result['metadata']
                }, status=status.HTTP_201_CREATED)
            else:
                return Response({
                    'success': False,
                    'error': result['error']
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'error': f'Error processing file: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UploadedFilesView(APIView):
    """
    API endpoint for managing uploaded files
    """
    
    def get(self, request):
        """List all uploaded files"""
        try:
            files = UploadedFile.objects.all().order_by('-uploaded_at')
            
            files_data = []
            for file in files:
                file_data = {
                    'id': file.id,
                    'original_filename': file.original_filename,
                    'uploaded_at': file.uploaded_at,
                    'file_size': file.file_size,
                    'processed': file.processed,
                    'sheet_count': file.sheet_count,
                    'row_count': file.row_count,
                    'column_count': file.column_count,
                    'processing_errors': file.processing_errors,
                    'sheets': []
                }
                
                # Get sheets for this file
                sheets = FileSheet.objects.filter(uploaded_file=file)
                for sheet in sheets:
                    file_data['sheets'].append({
                        'sheet_name': sheet.sheet_name,
                        'table_name': sheet.table_name,
                        'row_count': sheet.row_count,
                        'column_count': sheet.column_count
                    })
                
                files_data.append(file_data)
            
            return Response(files_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error retrieving uploaded files: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request, file_id):
        """Delete an uploaded file and its associated tables"""
        try:
            file = UploadedFile.objects.get(id=file_id)
            
            # Delete associated tables from database
            sheets = FileSheet.objects.filter(uploaded_file=file)
            with connection.cursor() as cursor:
                for sheet in sheets:
                    try:
                        cursor.execute(f'DROP TABLE IF EXISTS "{sheet.table_name}"')
                    except Exception as e:
                        print(f"Error dropping table {sheet.table_name}: {e}")
            
            # Delete file from filesystem
            if os.path.exists(file.file_path):
                os.remove(file.file_path)
            
            # Delete database records
            file.delete()
            
            return Response({
                'message': 'File and associated data deleted successfully'
            }, status=status.HTTP_200_OK)
            
        except UploadedFile.DoesNotExist:
            return Response({
                'error': 'File not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error deleting file: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
