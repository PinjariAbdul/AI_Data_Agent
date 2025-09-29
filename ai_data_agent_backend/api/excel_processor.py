import pandas as pd
import os
import re
import json
from datetime import datetime
from django.db import connection
from django.conf import settings
from .models import UploadedFile, FileSheet, DatabaseSchema
import logging
import chardet

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """
    Handles Excel file upload, processing, and conversion to database tables.
    Designed to handle any Excel file format with bad/inconsistent data formatting.
    """
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.xls', '.csv']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        
    def process_uploaded_file(self, uploaded_file, file_path):
        """
        Main method to process an uploaded Excel file.
        Returns: dict with processing results and metadata
        """
        try:
            # Create UploadedFile record
            file_record = UploadedFile.objects.create(
                file_name=self._generate_safe_filename(uploaded_file.name),
                original_filename=uploaded_file.name,
                file_path=file_path,
                file_size=uploaded_file.size
            )
            
            # Determine file type and process accordingly
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.csv':
                result = self._process_csv_file(file_path, file_record)
            elif file_extension in ['.xlsx', '.xls']:
                result = self._process_excel_file(file_path, file_record)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Update file record with processing results
            file_record.processed = True
            file_record.sheet_count = result['sheet_count']
            file_record.row_count = result['total_rows']
            file_record.column_count = result['total_columns']
            file_record.data_quality_summary = json.dumps(result['data_quality'])
            file_record.save()
            
            return {
                'success': True,
                'file_id': file_record.id,
                'tables_created': result['tables_created'],
                'metadata': result
            }
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            if 'file_record' in locals():
                file_record.processing_errors = str(e)
                file_record.save()
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _process_csv_file(self, file_path, file_record):
        """Process CSV file"""
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        
        # Try different separators
        separators = [',', ';', '\t', '|']
        df = None
        best_sep = None
        
        for sep in separators:
            try:
                # Try reading with current separator
                temp_df = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                   on_bad_lines='skip', low_memory=False)
                # Check if we got reasonable data
                if not temp_df.empty and (len(temp_df.columns) > 1 or (len(temp_df.columns) == 1 and len(temp_df) > 0)):
                    df = temp_df
                    best_sep = sep
                    break
            except Exception as e:
                logger.debug(f"Failed to parse CSV with separator '{sep}': {str(e)}")
                continue
        
        # If no separator worked, try with default pandas behavior
        if df is None:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', low_memory=False)
            except Exception as e:
                logger.error(f"Failed to parse CSV with default settings: {str(e)}")
        
        if df is None or df.empty:
            raise ValueError("Could not parse CSV file with any common separator or default settings")
        
        # Clean and process the dataframe
        df = self._clean_dataframe(df, "Sheet1")
        
        # Create table name
        table_name = self._generate_table_name(file_record.file_name, "Sheet1")
        
        # Create table in database
        self._create_table_from_dataframe(df, table_name)
        
        # Create FileSheet record
        FileSheet.objects.create(
            uploaded_file=file_record,
            sheet_name="Sheet1",
            original_sheet_name="CSV_Data",
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns)
        )
        
        # Update database schema metadata
        self._update_schema_metadata(table_name, df)
        
        return {
            'sheet_count': 1,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'tables_created': [table_name],
            'data_quality': self._analyze_data_quality(df)
        }
    
    def _process_excel_file(self, file_path, file_record):
        """Process Excel file with multiple sheets"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            tables_created = []
            total_rows = 0
            total_columns = 0
            all_data_quality = {}
            
            for i, sheet_name in enumerate(sheet_names):
                try:
                    # Read sheet with error handling
                    df = pd.read_excel(file_path, sheet_name=sheet_name, 
                                     header=None, dtype=str)
                    
                    if df.empty:
                        continue
                    
                    # Auto-detect header row
                    header_row = self._detect_header_row(df)
                    
                    # Re-read with proper header
                    if header_row > 0:
                        df = pd.read_excel(file_path, sheet_name=sheet_name,
                                         header=header_row, dtype=str)
                    else:
                        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
                    
                    # Clean and process the dataframe
                    df = self._clean_dataframe(df, sheet_name)
                    
                    if df.empty:
                        continue
                    
                    # Generate table name
                    clean_sheet_name = self._clean_sheet_name(sheet_name)
                    table_name = self._generate_table_name(file_record.file_name, clean_sheet_name)
                    
                    # Create table in database
                    self._create_table_from_dataframe(df, table_name)
                    
                    # Create FileSheet record
                    FileSheet.objects.create(
                        uploaded_file=file_record,
                        sheet_name=clean_sheet_name,
                        original_sheet_name=sheet_name,
                        table_name=table_name,
                        row_count=len(df),
                        column_count=len(df.columns),
                        header_row=header_row
                    )
                    
                    # Update database schema metadata
                    self._update_schema_metadata(table_name, df)
                    
                    tables_created.append(table_name)
                    total_rows += len(df)
                    total_columns += len(df.columns)
                    all_data_quality[table_name] = self._analyze_data_quality(df)
                    
                except Exception as e:
                    logger.warning(f"Error processing sheet {sheet_name}: {str(e)}")
                    continue
            
            return {
                'sheet_count': len(tables_created),
                'total_rows': total_rows,
                'total_columns': total_columns,
                'tables_created': tables_created,
                'data_quality': all_data_quality
            }
            
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
    
    def _detect_encoding(self, file_path):
        """Detect file encoding for CSV files"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _detect_header_row(self, df):
        """Auto-detect which row contains the header"""
        if df.empty:
            return 0
        
        # Look for first row that looks like headers
        for i in range(min(5, len(df))):  # Check first 5 rows
            row = df.iloc[i]
            # Count non-null values and check for text patterns
            non_null_count = row.notna().sum()
            text_count = sum(1 for val in row if isinstance(val, str) and len(str(val).strip()) > 0)
            
            # If most values are non-null and contain text, likely a header
            if non_null_count > len(row) * 0.5 and text_count > len(row) * 0.3:
                return i
        
        return 0
    
    def _clean_dataframe(self, df, sheet_name):
        """Clean and standardize dataframe"""
        if df.empty:
            return df
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            return df
        
        # Clean column names
        df.columns = [self._clean_column_name(str(col), i) for i, col in enumerate(df.columns)]
        
        # Handle duplicate column names
        seen_columns = {}
        new_columns = []
        for col in df.columns:
            if col in seen_columns:
                seen_columns[col] += 1
                new_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                new_columns.append(col)
        df.columns = new_columns
        
        # Basic data cleaning
        df = df.replace('', None)  # Replace empty strings with None
        df = df.where(pd.notnull(df), None)  # Replace NaN with None
        
        return df
    
    def _clean_column_name(self, col_name, index):
        """Clean and standardize column names"""
        if pd.isna(col_name) or str(col_name).strip() == '' or str(col_name).startswith('Unnamed'):
            return f"column_{index + 1}"
        
        # Clean the column name
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(col_name).strip())
        cleaned = re.sub(r'_+', '_', cleaned)  # Replace multiple underscores
        cleaned = cleaned.strip('_')  # Remove leading/trailing underscores
        
        # Ensure it starts with letter or underscore
        if cleaned and not cleaned[0].isalpha() and cleaned[0] != '_':
            cleaned = f"col_{cleaned}"
        
        # If still empty, use default
        if not cleaned:
            cleaned = f"column_{index + 1}"
        
        return cleaned.lower()
    
    def _clean_sheet_name(self, sheet_name):
        """Clean sheet name for use in table naming"""
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(sheet_name))
        cleaned = re.sub(r'_+', '_', cleaned)
        cleaned = cleaned.strip('_')
        return cleaned.lower() if cleaned else 'sheet'
    
    def _generate_table_name(self, filename, sheet_name):
        """Generate a unique table name"""
        base_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(filename)[0])
        base_name = re.sub(r'_+', '_', base_name).strip('_').lower()
        
        if not base_name:
            base_name = 'uploaded_data'
        
        table_name = f"upload_{base_name}_{sheet_name}"
        
        # Ensure uniqueness by adding timestamp if needed
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"{table_name}_{timestamp}"
        
        return final_name[:63]  # Limit length for database compatibility
    
    def _generate_safe_filename(self, original_name):
        """Generate safe filename for storage"""
        name, ext = os.path.splitext(original_name)
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{safe_name}_{timestamp}{ext}"
    
    def _create_table_from_dataframe(self, df, table_name):
        """Create database table from dataframe"""
        if df.empty:
            raise ValueError("Cannot create table from empty dataframe")
        
        # Analyze column types
        columns_sql = []
        for col in df.columns:
            column_type = self._determine_column_type(df[col])
            columns_sql.append(f'"{col}" {column_type}')
        
        # Create table
        create_sql = f'''
        CREATE TABLE "{table_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(columns_sql)}
        )
        '''
        
        with connection.cursor() as cursor:
            cursor.execute(create_sql)
        
        # Insert data
        self._insert_dataframe_to_table(df, table_name)
    
    def _determine_column_type(self, series):
        """Determine appropriate SQL column type for a pandas series"""
        # Try to convert to numeric
        numeric_count = 0
        date_count = 0
        total_non_null = series.notna().sum()
        
        if total_non_null == 0:
            return "TEXT"
        
        for value in series.dropna():
            str_value = str(value).strip()
            
            # Check if it's numeric
            try:
                float(str_value)
                numeric_count += 1
                continue
            except:
                pass
            
            # Check if it's a date
            try:
                pd.to_datetime(str_value)
                date_count += 1
                continue
            except:
                pass
        
        # Determine type based on counts
        if numeric_count > total_non_null * 0.8:
            # Check if it's integer or float
            try:
                series_numeric = pd.to_numeric(series, errors='coerce')
                if series_numeric.notna().sum() > 0:
                    if (series_numeric % 1 == 0).all():
                        return "INTEGER"
                    else:
                        return "REAL"
            except:
                pass
            return "REAL"
        
        if date_count > total_non_null * 0.8:
            return "TEXT"  # Store dates as text for flexibility
        
        return "TEXT"
    
    def _insert_dataframe_to_table(self, df, table_name):
        """Insert dataframe data into database table"""
        if df.empty:
            return
        
        # Prepare data for insertion
        columns = [f'"{col}"' for col in df.columns]
        placeholders = ', '.join(['?' for _ in columns])
        
        insert_sql = f'INSERT INTO "{table_name}" ({", ".join(columns)}) VALUES ({placeholders})'
        
        # Convert dataframe to list of tuples
        data_tuples = []
        for _, row in df.iterrows():
            tuple_data = []
            for value in row:
                if pd.isna(value) or value == '':
                    tuple_data.append(None)
                else:
                    tuple_data.append(str(value))
            data_tuples.append(tuple(tuple_data))
        
        # Insert in batches
        batch_size = 1000
        with connection.cursor() as cursor:
            for i in range(0, len(data_tuples), batch_size):
                batch = data_tuples[i:i + batch_size]
                cursor.executemany(insert_sql, batch)
    
    def _update_schema_metadata(self, table_name, df):
        """Update DatabaseSchema metadata for AI agent"""
        for col in df.columns:
            # Analyze column to provide meaningful description
            sample_values = df[col].dropna().head(5).tolist()
            
            # Try to infer what this column represents
            actual_meaning = self._infer_column_meaning(col, sample_values)
            
            DatabaseSchema.objects.update_or_create(
                table_name=table_name,
                column_name=col,
                defaults={
                    'actual_meaning': actual_meaning,
                    'data_type': str(df[col].dtype),
                    'description': f"Column from uploaded Excel file",
                    'sample_values': json.dumps(sample_values),
                    'data_quality_issues': self._analyze_column_quality(df[col])
                }
            )
    
    def _infer_column_meaning(self, column_name, sample_values):
        """Try to infer what a column represents based on name and values"""
        col_lower = column_name.lower()
        
        # Common patterns
        if any(word in col_lower for word in ['name', 'nm', 'title']):
            return "Name/Title"
        elif any(word in col_lower for word in ['email', 'mail', 'em']):
            return "Email Address"
        elif any(word in col_lower for word in ['phone', 'ph', 'tel', 'mobile']):
            return "Phone Number"
        elif any(word in col_lower for word in ['addr', 'address']):
            return "Address"
        elif any(word in col_lower for word in ['city', 'cty']):
            return "City"
        elif any(word in col_lower for word in ['state', 'st', 'province']):
            return "State/Province"
        elif any(word in col_lower for word in ['zip', 'postal', 'code']):
            return "Postal Code"
        elif any(word in col_lower for word in ['country', 'cntry']):
            return "Country"
        elif any(word in col_lower for word in ['price', 'cost', 'amount', 'amt']):
            return "Price/Amount"
        elif any(word in col_lower for word in ['quantity', 'qty', 'count']):
            return "Quantity/Count"
        elif any(word in col_lower for word in ['date', 'dt', 'time']):
            return "Date/Time"
        elif any(word in col_lower for word in ['id', 'key']):
            return "Identifier"
        else:
            return column_name.replace('_', ' ').title()
    
    def _analyze_column_quality(self, series):
        """Analyze data quality issues for a column"""
        issues = []
        total_count = len(series)
        null_count = series.isna().sum()
        
        if null_count > 0:
            null_percentage = (null_count / total_count) * 100
            issues.append(f"Missing values: {null_percentage:.1f}%")
        
        # Check for duplicates
        duplicate_count = series.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Duplicate values: {duplicate_count}")
        
        return "; ".join(issues) if issues else "No major issues detected"
    
    def _analyze_data_quality(self, df):
        """Analyze overall data quality of dataframe"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'columns_with_all_nulls': df.columns[df.isna().all()].tolist(),
            'columns_with_high_nulls': df.columns[df.isna().mean() > 0.5].tolist()
        }