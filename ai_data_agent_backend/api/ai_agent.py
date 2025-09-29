import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from django.conf import settings
from django.db import connection
from .models import *
import re
from datetime import datetime
import base64
import io
from typing import Dict, List, Tuple, Any, Optional
import requests  # For alternative APIs
from .offline_ai import OfflineAIAgent
from .advanced_analytics import AdvancedAnalyticsEngine
from .query_optimizer import IntelligentQueryOptimizer

class SQLQueryAgent:
    """
    AI Agent that generates SQL queries from natural language questions
    and handles complex database schemas with poor naming conventions.
    """
    
    def __init__(self):
        # Get AI configuration from settings
        ai_config = getattr(settings, 'AI_CONFIG', {})
        primary_provider = ai_config.get('PRIMARY_PROVIDER', 'groq')
        
        # Load API keys from settings
        self.groq_api_key = getattr(settings, 'GROQ_API_KEY', None)
        self.together_api_key = getattr(settings, 'TOGETHER_API_KEY', None)
        self.huggingface_token = getattr(settings, 'HUGGINGFACE_TOKEN', None)
        self.google_api_key = getattr(settings, 'GOOGLE_API_KEY', None)
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        
        # PRIORITIZE GROQ: Check Groq first regardless of other keys
        if self.groq_api_key and self.groq_api_key.strip():
            self.ai_provider = "groq"
            self.demo_mode = False
            self.client = None
            print("🚀 Using Groq as primary AI provider (Fast & Free!)")
            print(f"✅ Groq API Key loaded: {self.groq_api_key[:10]}...")
        elif api_key and api_key not in ['your_openai_api_key_here', 'sk-test-key-replace-with-your-actual-openai-api-key-here']:
            self.demo_mode = False
            self.ai_provider = "openai"
            try:
                openai.api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"⚠️ OpenAI failed, switching to free alternatives: {str(e)}")
                self.client = None
                self.ai_provider = "free_alternatives"
        else:
            # Use free alternatives or offline mode
            self.demo_mode = False
            self.client = None
            if any([self.together_api_key, self.huggingface_token, self.google_api_key]):
                self.ai_provider = "free_alternatives"
                print("🆓 Using free AI alternatives")
            else:
                self.ai_provider = "offline"
                print("💻 Using offline AI system (no APIs needed)")
        
        # Set preferred model with fallback options
        self.preferred_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        self.current_model = self.preferred_models[0]  # Start with most accessible model
        
        # Free AI alternatives configuration
        self.free_models = {
            "groq": "llama-3.1-8b-instant",  # Updated to current supported model
            "huggingface": "microsoft/DialoGPT-medium",
            "together": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google": "gemini-pro"
        }
        
        # Initialize offline AI as final fallback
        self.offline_ai = OfflineAIAgent()
        
        # Initialize advanced analytics engine (with error handling)
        try:
            self.advanced_analytics = AdvancedAnalyticsEngine()
        except Exception as e:
            print(f"⚠️ Advanced analytics disabled: {str(e)}")
            self.advanced_analytics = None
        
        # Initialize intelligent query optimizer (with error handling)
        try:
            self.query_optimizer = IntelligentQueryOptimizer()
        except Exception as e:
            print(f"⚠️ Query optimizer disabled: {str(e)}")
            self.query_optimizer = None
        
        self.schema_info = self._build_schema_context()
    
    def _build_schema_context(self, uploaded_tables: Optional[List[str]] = None) -> str:
        """
        Build a comprehensive context about the database schema including
        poor naming conventions and actual column meanings.
        """
        # Base schema context for default tables
        base_schema_context = '''
        DATABASE SCHEMA INFORMATION:
        
        This database has very poor naming conventions and unclear column names. Here's what each table and column actually means:
        
        1. TABLE: cust_data (Customer Data)
           - id: Customer ID (primary key)
           - nm: Customer Name
           - emladdr: Email Address
           - ph_num: Phone Number
           - addr_ln1: Address Line 1
           - addr_ln2: Address Line 2
           - cty: City
           - st: State
           - zip_cd: Zip Code
           - cntry: Country
           - reg_dt: Registration Date
           - lst_login: Last Login Date
           - is_actv: Is Active (boolean)
           - cust_type: Customer Type (premium, regular, etc.)
        
        2. TABLE: prod_tbl (Product Table)
           - id: Product ID (primary key)
           - col1: Product Name
           - col2: Product Description
           - col3: Price
           - col4: Category
           - col5: Brand
           - col6: Stock Quantity
           - col7: SKU (Stock Keeping Unit)
           - col8: Is Available (boolean)
           - col9: Created Date
           - col10: Supplier Information
        
        3. TABLE: Order_History (Order History)
           - order_ID: Order ID (primary key)
           - customer_ref_id: Customer Reference ID (foreign key to cust_data.id)
           - order_date_time: Order Date and Time
           - total_amt: Total Amount
           - order_status: Order Status (pending, completed, cancelled, etc.)
           - payment_method: Payment Method
           - shipping_addr: Shipping Address
           - delivery_date: Delivery Date
           - discount_applied: Discount Applied
           - tax_amount: Tax Amount
        
        4. TABLE: orderitems (Order Items)
           - item_id: Item ID (primary key)
           - ORDER_REF: Order Reference (foreign key to Order_History.order_ID)
           - product_ref: Product Reference (foreign key to prod_tbl.id)
           - qty: Quantity
           - unit_price: Unit Price
           - total_price: Total Price
           - item_discount: Item Discount
        
        5. TABLE: sales_data_2023 (Sales Data for 2023)
           - id: Record ID (primary key)
           - a1: Region
           - b2: Sales Representative
           - c3: Revenue
           - d4: Units Sold
           - e5: Sale Date
           - f6: Product Category
           - g7: Commission
           - h8: Quarter
        
        6. TABLE: EMP_RECORDS (Employee Records)
           - emp_id: Employee ID (primary key)
           - first_name: First Name
           - last_name: Last Name
           - email: Email
           - department: Department
           - position: Position
           - salary: Salary
           - hire_date: Hire Date
           - manager_id: Manager ID
           - is_active: Is Active (boolean)
        
        COMMON DATA QUALITY ISSUES:
        - Missing values in many fields
        - Inconsistent date formats
        - Mixed case in text fields
        - Duplicate records possible
        - Some foreign key relationships may have orphaned records
        '''
        
        # If we have uploaded tables, add information about them
        if uploaded_tables:
            uploaded_context = f"""
            \nADDITIONAL UPLOADED TABLES:
            The following tables were recently uploaded by the user and should be considered for queries:
            
            """
            for table in uploaded_tables:
                uploaded_context += f"- TABLE: {table}\n"
            
            return base_schema_context + uploaded_context
        
        return base_schema_context
    
    def _try_api_call_with_fallback(self, messages, temperature=0.1):
        """
        Try API call with model fallback if the current model is not available.
        """
        for model in self.preferred_models:
            try:
                if self.client and hasattr(self.client, 'chat'):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                    self.current_model = model  # Update current working model
                    return response
                else:
                    # No client available, continue to next model
                    continue
            except Exception as e:
                error_str = str(e)
                # Handle different types of API errors
                if "does not exist" in error_str or "do not have access" in error_str:
                    print(f"⚠️  Model {model} not available, trying next...")
                    continue
                elif "429" in error_str or "insufficient_quota" in error_str or "quota" in error_str:
                    print(f"⚠️  OpenAI API quota exceeded. Switching to demo mode.")
                    print(f"💡 To restore full AI features, add credits at: https://platform.openai.com/account/billing")
                    # Switch to demo mode for this session
                    self.demo_mode = True
                    raise Exception("API_QUOTA_EXCEEDED")
                elif "rate_limit" in error_str:
                    print(f"⚠️  Rate limit exceeded for {model}, trying next...")
                    continue
                else:
                    # Other error, re-raise
                    raise e
        
        # If all models failed, return None
        return None
    
    def _call_free_ai_api(self, prompt: str, task_type: str = "general") -> str:
        """
        Call free AI APIs as alternatives to OpenAI.
        """
        print("🆓 Trying free AI alternatives...")
        
        # Try Groq API (fastest free option)
        if self.groq_api_key:
            try:
                print("🚀 Trying Groq API...")
                response = self._call_groq_api(prompt, task_type)
                if response:
                    print("✅ Groq succeeded!")
                    return response
            except Exception as e:
                print(f"❌ Groq API failed: {str(e)}")
        else:
            print("⚠️ No Groq API key configured")
        
        # Try Together AI
        if self.together_api_key:
            try:
                print("🤖 Trying Together AI...")
                response = self._call_together_api(prompt, task_type)
                if response:
                    print("✅ Together AI succeeded!")
                    return response
            except Exception as e:
                print(f"❌ Together AI failed: {str(e)}")
        else:
            print("⚠️ No Together AI API key configured")
        
        # Try Hugging Face Inference API (Free)
        if self.huggingface_token:
            try:
                print("🤗 Trying Hugging Face API...")
                response = self._generate_sql_offline(prompt, task_type)  # Fallback to offline for now
                if response:
                    print("✅ Hugging Face (offline fallback) succeeded!")
                    return response
            except Exception as e:
                print(f"❌ Hugging Face API failed: {str(e)}")
        else:
            print("⚠️ No Hugging Face token configured")
        
        # Try Google Gemini
        if self.google_api_key:
            try:
                print("🧠 Trying Google Gemini...")
                response = self._call_google_api(prompt, task_type)
                if response:
                    print("✅ Google Gemini succeeded!")
                    return response
            except Exception as e:
                print(f"❌ Google Gemini failed: {str(e)}")
        else:
            print("⚠️ No Google API key configured")
        
        # Use offline AI system as final fallback
        try:
            print("💻 Using offline AI system...")
            if task_type == "sql_generation":
                response = self.offline_ai.generate_sql(prompt)
                if response:
                    print("✅ Offline AI succeeded!")
                    return response
            response = self._generate_sql_offline(prompt, task_type)
            return response
        except Exception as e:
            print(f"❌ Offline generation failed: {str(e)}")
        
        # Ultimate fallback to enhanced demo mode
        return self._enhanced_demo_response(prompt, task_type)
    
    def _call_groq_api(self, prompt: str, task_type: str = "general") -> str:
        """
        Call Groq API for fast inference.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a SQL expert. Generate only SQL queries, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                "model": "llama-3.1-8b-instant",  # Updated model
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    def _call_together_api(self, prompt: str, task_type: str = "general") -> str:
        """
        Call Together AI API.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You are a SQL expert. Generate only SQL queries, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"Together API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Together API error: {str(e)}")
    
    def _call_google_api(self, prompt: str, task_type: str = "general") -> str:
        """
        Call Google Gemini API.
        """
        try:
            headers = {
                "x-goog-api-key": self.google_api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"You are a SQL expert. Generate only SQL queries, no explanations.\n\n{prompt}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000
                }
            }
            
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['candidates'][0]['content']['parts'][0]['text']
                if content:
                    return content.strip()
                else:
                    return ""
            else:
                raise Exception(f"Google API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")
    
    def _detect_relevant_tables(self, question: str, uploaded_tables: Optional[List[str]] = None) -> List[str]:
        """
        Detect which tables are relevant based on question keywords.
        """
        if uploaded_tables is None:
            uploaded_tables = []
            
        question_lower = question.lower()
        tables = []
        
        # If we have uploaded tables, prioritize them
        if uploaded_tables:
            # Check for keywords in the question that match uploaded table names
            for table in uploaded_tables:
                table_lower = table.lower()
                if any(word in question_lower for word in ['customer', 'client', 'user']) and 'customer' in table_lower:
                    tables.append(table)
                elif any(word in question_lower for word in ['product', 'item', 'inventory']) and 'product' in table_lower:
                    tables.append(table)
                elif any(word in question_lower for word in ['order', 'purchase', 'transaction']) and ('order' in table_lower or 'transaction' in table_lower):
                    tables.append(table)
                elif any(word in question_lower for word in ['sales', 'revenue', 'commission']) and ('sales' in table_lower or 'revenue' in table_lower):
                    tables.append(table)
                elif any(word in question_lower for word in ['employee', 'staff', 'worker']) and ('employee' in table_lower or 'staff' in table_lower):
                    tables.append(table)
        
        # If no tables matched uploaded tables, use default schema
        if not tables:
            if any(word in question_lower for word in ['customer', 'client', 'user']):
                tables.append('cust_data')
            if any(word in question_lower for word in ['product', 'item', 'inventory']):
                tables.append('prod_tbl')
            if any(word in question_lower for word in ['order', 'purchase', 'transaction']):
                tables.append('Order_History')
                tables.append('orderitems')
            if any(word in question_lower for word in ['sales', 'revenue', 'commission']):
                tables.append('sales_data_2023')
            if any(word in question_lower for word in ['employee', 'staff', 'worker']):
                tables.append('EMP_RECORDS')
        
        # If still no tables, return a default set based on question type
        if not tables:
            # Try to determine most relevant table based on question focus
            if any(word in question_lower for word in ['customer', 'client', 'user']):
                tables.append('cust_data')
            elif any(word in question_lower for word in ['product', 'item', 'inventory']):
                tables.append('prod_tbl')
            elif any(word in question_lower for word in ['order', 'purchase', 'transaction']):
                tables.append('Order_History')
            elif any(word in question_lower for word in ['sales', 'revenue', 'commission']):
                tables.append('sales_data_2023')
            elif any(word in question_lower for word in ['employee', 'staff', 'worker']):
                tables.append('EMP_RECORDS')
        
        return tables if tables else ['cust_data']  # Default fallback
    
    def _detect_analysis_type(self, question: str) -> str:
        """
        Detect the type of analysis needed.
        """
        question_lower = question.lower()
        
        # More specific detection for different analysis types
        if any(word in question_lower for word in ['count', 'total', 'sum', 'average', 'avg', 'mean', 'median', 'how many']):
            return 'aggregation'
        elif any(word in question_lower for word in ['top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'most', 'least', 'rank']):
            return 'ranking'
        elif any(word in question_lower for word in ['group', 'category', 'department', 'region', 'by', 'per', 'each']):
            return 'grouping'
        elif any(word in question_lower for word in ['trend', 'over time', 'monthly', 'yearly', 'quarterly', 'daily', 'growth', 'change over time']):
            return 'time_series'
        elif any(word in question_lower for word in ['join', 'relationship', 'connect', 'link', 'combine', 'together']):
            return 'joining'
        elif any(word in question_lower for word in ['filter', 'where', 'which', 'find', 'show me']):
            return 'filtering'
        else:
            return 'basic_query'
    
    def _suggest_visualization(self, question: str, analysis_type: str) -> str:
        """
        Suggest appropriate visualization type.
        """
        question_lower = question.lower()
        
        # More specific visualization suggestions based on question and analysis type
        if any(word in question_lower for word in ['trend', 'over time', 'monthly', 'yearly', 'growth', 'change over time']):
            return 'line_chart'
        elif any(word in question_lower for word in ['distribution', 'breakdown', 'percentage', 'proportion', 'share']):
            return 'pie_chart'
        elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'top', 'bottom', 'rank', 'most', 'least']):
            return 'bar_chart'
        elif any(word in question_lower for word in ['correlation', 'relationship', 'scatter', 'related to', 'connected to']):
            return 'scatter_plot'
        elif analysis_type == 'time_series':
            return 'line_chart'
        elif analysis_type == 'ranking':
            return 'bar_chart'
        elif analysis_type == 'grouping':
            # For grouping, choose based on data type
            if any(word in question_lower for word in ['distribution', 'percentage', 'proportion']):
                return 'pie_chart'
            else:
                return 'bar_chart'
        else:
            return 'table'
    
    def _generate_sql_offline(self, prompt: str, task_type: str) -> str:
        """
        Generate SQL using pattern matching and rules (no AI API needed).
        """
        prompt_lower = prompt.lower()
        
        # Enhanced pattern matching for SQL generation
        if task_type == "sql_generation":
            if "customer" in prompt_lower and "count" in prompt_lower:
                return "SELECT COUNT(*) as customer_count FROM cust_data WHERE is_actv = 1;"
            elif "customer" in prompt_lower and "california" in prompt_lower:
                return "SELECT nm, emladdr FROM cust_data WHERE st = 'CA' AND is_actv = 1;"
            elif "product" in prompt_lower and "top" in prompt_lower:
                return "SELECT col1 as product_name, SUM(oi.total_price) as revenue FROM prod_tbl p JOIN orderitems oi ON p.id = oi.product_ref GROUP BY p.id, col1 ORDER BY revenue DESC LIMIT 10;"
            elif "sales" in prompt_lower and "region" in prompt_lower:
                return "SELECT a1 as region, SUM(c3) as total_sales FROM sales_data_2023 GROUP BY a1 ORDER BY total_sales DESC;"
            elif "employee" in prompt_lower and "department" in prompt_lower:
                return "SELECT department, COUNT(*) as employee_count FROM EMP_RECORDS WHERE is_active = 1 GROUP BY department;"
            elif "order" in prompt_lower and "average" in prompt_lower:
                return "SELECT AVG(total_amt) as average_order_value FROM Order_History WHERE order_status = 'completed';"
            else:
                return "SELECT COUNT(*) as total_records FROM cust_data;"
        
        elif task_type == "analysis":
            return '{"intent": "Business query analysis", "required_tables": ["cust_data"], "analysis_type": "basic_query", "suggested_visualization": "table"}'
        
        elif task_type == "explanation":
            return "Based on the query results, here's what the data shows: The analysis provides insights into your business metrics. The results indicate patterns in your data that can help inform business decisions."
        
        return "Query processed successfully using offline analysis."
    
    def _enhanced_demo_response(self, prompt: str, task_type: str) -> str:
        """
        Enhanced demo responses with better pattern matching.
        """
        if task_type == "sql_generation":
            return self._generate_sql_offline(prompt, task_type)
        elif task_type == "analysis":
            return '{"intent": "Demo analysis", "required_tables": ["cust_data", "prod_tbl"], "analysis_type": "basic_query", "suggested_visualization": "table"}'
        else:
            return "This is a demo response. For full AI capabilities, consider upgrading or using free alternatives like Hugging Face."
    
    def analyze_question(self, question: str, uploaded_tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the natural language question to understand intent
        and identify relevant tables/columns.
        """
        if uploaded_tables is None:
            uploaded_tables = []
        
        try:
            # Use Groq if it's the primary provider
            if self.ai_provider == "groq":
                try:
                    return self._analyze_question_with_groq(question, uploaded_tables)
                except Exception as e:
                    print(f"⚠️ Groq analysis failed: {str(e)}")
                    # Fall back to enhanced demo mode
                    pass
            
            # Enhanced demo mode fallback
            analysis_type = self._detect_analysis_type(question)
            return {
                "intent": f"Smart analysis for: {question}",
                "required_tables": self._detect_relevant_tables(question, uploaded_tables),
                "analysis_type": analysis_type,
                "suggested_visualization": self._suggest_visualization(question, analysis_type)
            }
        except Exception as e:
            # Ultimate fallback
            return {
                "intent": f"Fallback analysis for: {question}",
                "required_tables": ["cust_data"],
                "analysis_type": "basic_query",
                "suggested_visualization": "table"
            }
    
    def _analyze_question_with_groq(self, question: str, uploaded_tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze question using Groq API.
        """
        if uploaded_tables is None:
            uploaded_tables = []
            
        analysis_prompt = f'''Analyze this business question and identify:
1. What information is being asked for
2. Which tables are likely needed
3. What type of analysis is required (aggregation, filtering, joining, etc.)
4. What visualization would be most appropriate

Question: {question}

{self.schema_info}

Return your analysis as JSON with keys: intent, required_tables, analysis_type, suggested_visualization'''
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a database analyst expert at understanding business questions and mapping them to database queries. Return only valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                "model": "llama-3.1-8b-instant",  # Updated model
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content'].strip()
                # Extract JSON from the response
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON from Groq response")
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Groq analysis failed: {str(e)}")
        
        # This shouldn't be reached, but add fallback for safety
        return {
            "intent": f"Fallback analysis for: {question}",
            "required_tables": self._detect_relevant_tables(question),
            "analysis_type": self._detect_analysis_type(question),
            "suggested_visualization": self._suggest_visualization(question)
        }

    
    def generate_sql_query(self, question: str, context: str = "") -> Tuple[str, float]:
        """
        Generate SQL query from natural language question.
        Returns tuple of (sql_query, confidence_score)
        """
        
        # Primary: Use Groq if available
        if self.ai_provider == "groq" and self.groq_api_key:
            try:
                print("🚀 Using Groq AI for SQL generation...")
                analysis = self.analyze_question(question)
                sql_query = self._generate_sql_with_groq(question, context, analysis)
                confidence = self._calculate_confidence(sql_query, analysis)
                return sql_query, confidence
            except Exception as e:
                print(f"⚠️ Groq failed, falling back: {str(e)}")
        
        # Fallback to existing logic
        if self.demo_mode:
            # Demo mode fallback with simple hardcoded queries
            question_lower = question.lower()
            if "customer" in question_lower and "count" in question_lower:
                return "SELECT COUNT(*) as customer_count FROM cust_data WHERE is_actv = 1;", 0.8
            elif "sales" in question_lower and "region" in question_lower:
                return "SELECT a1 as region, SUM(c3) as total_sales FROM sales_data_2023 GROUP BY a1 ORDER BY total_sales DESC;", 0.8
            elif "product" in question_lower and "top" in question_lower:
                return "SELECT col1 as product_name, SUM(oi.total_price) as revenue FROM prod_tbl p JOIN orderitems oi ON p.id = oi.product_ref GROUP BY p.id, col1 ORDER BY revenue DESC LIMIT 10;", 0.7
            else:
                return "SELECT COUNT(*) as total_records FROM cust_data;", 0.5
        
        analysis = self.analyze_question(question)
        
        sql_prompt = f'''
        Generate a SINGLE, FOCUSED SQL query to answer this business question. 
        The database has poor naming conventions, so use the schema information provided.
        
        Question: {question}
        Context: {context}
        
        Analysis: {json.dumps(analysis, indent=2)}
        
        {self.schema_info}
        
        IMPORTANT RULES:
        1. Generate ONLY ONE query - DO NOT use UNION or UNION ALL to combine multiple queries
        2. Focus on the specific question being asked
        3. Use the exact table and column names as specified in the schema
        4. Handle potential NULL values appropriately
        5. Consider data quality issues (duplicates, missing data)
        6. Use appropriate JOINs for related data
        7. Add appropriate WHERE clauses to filter out invalid data
        8. Use LIMIT when appropriate to prevent overwhelming results
        9. DO NOT include multiple unrelated queries combined with UNION
        
        Return ONLY the SQL query, no explanations or markdown formatting.
        '''
        
        try:
            response = self._try_api_call_with_fallback([
                {"role": "system", "content": "You are a SQL expert who writes precise, focused queries for databases with poor schema design. Generate ONLY ONE query to answer the specific question. DO NOT use UNION statements. Return only the SQL query."},
                {"role": "user", "content": sql_prompt}
            ], temperature=0.1)
            
            # Check if response is valid before accessing its attributes
            if response is not None and hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content') and choice.message.content:
                    sql_query = choice.message.content.strip()
                else:
                    sql_query = "-- Error: No content in response"
            else:
                sql_query = "-- Error: No valid response from AI model"
            
            # Remove any markdown formatting
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            
            # Simple confidence scoring based on query complexity and analysis
            confidence = self._calculate_confidence(sql_query, analysis)
            
            return sql_query, confidence
            
        except Exception as e:
            error_str = str(e)
            if "API_QUOTA_EXCEEDED" in error_str:
                # Fall back to demo mode queries
                print(f"💰 Switching to demo mode due to quota limits")
                question_lower = question.lower()
                if "customer" in question_lower and "count" in question_lower:
                    return "SELECT COUNT(*) as customer_count FROM cust_data WHERE is_actv = 1;", 0.8
                elif "sales" in question_lower and "region" in question_lower:
                    return "SELECT a1 as region, SUM(c3) as total_sales FROM sales_data_2023 GROUP BY a1 ORDER BY total_sales DESC;", 0.8
                elif "product" in question_lower and "top" in question_lower:
                    return "SELECT col1 as product_name, SUM(oi.total_price) as revenue FROM prod_tbl p JOIN orderitems oi ON p.id = oi.product_ref GROUP BY p.id, col1 ORDER BY revenue DESC LIMIT 10;", 0.7
                else:
                    return "SELECT COUNT(*) as total_records FROM cust_data;", 0.5
            return f"-- Error generating query: {str(e)}", 0.0
    
    def _generate_sql_with_groq(self, question: str, context: str, analysis: Dict) -> str:
        """
        Generate SQL query using Groq API specifically.
        """
        sql_prompt = f'''Generate a SINGLE, FOCUSED SQL query to answer this business question. The database has poor naming conventions, so use the schema information provided.

Question: {question}
Context: {context}

Analysis: {json.dumps(analysis, indent=2)}

{self.schema_info}

IMPORTANT RULES:
1. Generate ONLY ONE query - DO NOT use UNION or UNION ALL to combine multiple queries
2. Focus on the specific question being asked
3. Use the exact table and column names as specified in the schema
4. Handle potential NULL values appropriately
5. Consider data quality issues (duplicates, missing data)
6. Use appropriate JOINs for related data
7. Add appropriate WHERE clauses to filter out invalid data
8. Use LIMIT when appropriate to prevent overwhelming results
9. DO NOT include multiple unrelated queries combined with UNION

Return ONLY the SQL query, no explanations or markdown formatting.'''
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a SQL expert who writes precise, focused queries for databases with poor schema design. Generate ONLY ONE query to answer the specific question. DO NOT use UNION statements. Return only the SQL query."},
                    {"role": "user", "content": sql_prompt}
                ],
                "model": "llama-3.1-8b-instant",  # Updated model
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                sql_query = result['choices'][0]['message']['content'].strip()
                # Remove any markdown formatting
                sql_query = re.sub(r'```sql\n?', '', sql_query)
                sql_query = re.sub(r'```\n?', '', sql_query)
                return sql_query
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Groq SQL generation failed: {str(e)}")
    
    def _calculate_confidence(self, sql_query: str, analysis: Dict) -> float:
        """
        Calculate confidence score for the generated SQL query.
        """
        # Handle case where sql_query might be None or empty
        if not sql_query or not isinstance(sql_query, str):
            return 0.0
            
        confidence = 0.5  # Base confidence
        
        # Increase confidence for well-formed queries
        if "SELECT" in sql_query.upper() and "FROM" in sql_query.upper():
            confidence += 0.2
        
        # Increase confidence if required tables are mentioned
        required_tables = analysis.get('required_tables', [])
        if required_tables:
            table_matches = 0
            for table in required_tables:
                if table.lower() in sql_query.lower():
                    table_matches += 1
            # Add confidence based on percentage of required tables found
            confidence += 0.1 * (table_matches / len(required_tables))
        
        # Decrease confidence for overly complex queries
        if sql_query.count('UNION') > 2:  # Too many unions
            confidence -= 0.3
            
        # Decrease confidence for simple or potentially problematic queries
        if sql_query.count('\n') < 2:  # Very simple query
            confidence -= 0.1
        
        if "--" in sql_query:  # Contains error comments
            confidence = 0.0
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, confidence))
    
    def execute_query(self, sql_query: str) -> Tuple[List[Dict], str]:
        """
        Execute SQL query and return results.
        Returns tuple of (results, error_message)
        """
        try:
            # Validate SQL query
            if not sql_query or not isinstance(sql_query, str):
                return [], "Invalid SQL query"
            
            # Clean the query
            sql_query = sql_query.strip()
            if not sql_query:
                return [], "Empty SQL query"
            
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                if cursor.description:  # SELECT query
                    columns = [col[0] for col in cursor.description]
                    results = [
                        dict(zip(columns, row))
                        for row in cursor.fetchall()
                    ]
                    return results, ""
                else:  # INSERT/UPDATE/DELETE query
                    return [], "Query executed successfully (no results returned)"
        except Exception as e:
            error_msg = str(e)
            print(f"❌ SQL execution error: {error_msg}")
            print(f"   Query: {sql_query}")
            return [], error_msg
    
    def generate_natural_language_answer(self, question: str, sql_query: str, results: List[Dict]) -> str:
        """
        Generate natural language answer from query results.
        """
        if not results:
            return "No data found matching your query. This could be due to the data quality issues in the database or the specific criteria in your question."
        
        try:
            # Limit results for processing
            sample_results = results[:10]  # Process only first 10 results for efficiency
            
            # Prepare data for AI analysis
            data_summary = {
                "question": question,
                "sql_query": sql_query,
                "result_count": len(results),
                "sample_data": sample_results
            }
            
            # Try to use Groq for answer generation
            if self.ai_provider == "groq" and self.groq_api_key:
                try:
                    answer_prompt = f'''Based on the following query results, provide a clear, natural language answer to the business question.
                    
                    Question: {question}
                    Results: {json.dumps(data_summary, indent=2, default=str)}
                    
                    Provide a concise, business-friendly answer that explains what the data shows.
                    Do not mention technical details like SQL queries or table names.
                    Focus on the insights and implications of the data.
                    '''
                    
                    headers = {
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "messages": [
                            {"role": "system", "content": "You are a business analyst who explains data insights in clear, simple language. Focus on what the data means for business decisions."},
                            {"role": "user", "content": answer_prompt}
                        ],
                        "model": "llama-3.1-8b-instant",
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                    
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result['choices'][0]['message']['content'].strip()
                        if answer:
                            return answer
                except Exception as e:
                    print(f"⚠️ Groq answer generation failed: {str(e)}")
            
            # Fallback to pattern-based responses
            if len(results) == 1 and len(results[0]) == 1:
                # Single value result
                value = list(results[0].values())[0]
                return f"Based on the data analysis, the answer to your question is: {value}"
            elif len(results) <= 5:
                # Small result set
                return f"Based on the data analysis, there are {len(results)} results that match your query. Here are the key findings: {json.dumps(results, default=str)}"
            else:
                # Large result set
                return f"Based on the data analysis, there are {len(results)} results that match your query. The system has identified key patterns and insights in this data set that can inform business decisions."
        except Exception as e:
            print(f"❌ Error in natural language answer generation: {str(e)}")
            # Ultimate fallback
            return f"Based on the analysis of your question \"{question}\", the system found {len(results) if results else 0} matching records. Please review the data results directly for detailed insights."
    
    def generate_visualizations(self, results: List[Dict], suggested_viz: str = "table") -> Dict[str, Any]:
        """
        Generate appropriate visualizations based on data and suggested type.
        Returns a dictionary of visualization data.
        """
        if not results:
            return {"table": [], "message": "No data available for visualization"}
        
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(results)
            
            # Handle case where there's only one row
            if len(df) == 1:
                # For single row results, create a simple table visualization
                return {
                    "table": df.to_dict('records')
                }
            
            # For multiple rows, create appropriate visualizations
            visualizations = {}
            
            # Always generate a table view
            visualizations["table"] = df.to_dict('records')
            
            # Generate chart based on suggested type
            try:
                if suggested_viz == "line_chart" and len(df.columns) >= 2:
                    visualizations["line_chart"] = self._create_line_chart(df)
                elif suggested_viz == "bar_chart" and len(df.columns) >= 2:
                    visualizations["bar_chart"] = self._create_bar_chart(df)
                elif suggested_viz == "pie_chart" and len(df.columns) >= 2:
                    visualizations["pie_chart"] = self._create_pie_chart(df)
                elif suggested_viz == "scatter_plot" and len(df.columns) >= 2:
                    visualizations["scatter_plot"] = self._create_scatter_plot(df)
            except Exception as chart_error:
                print(f"⚠️ Chart generation failed: {chart_error}")
                visualizations["chart_error"] = f"Chart generation failed: {str(chart_error)}"
            
            # If no specific chart was suggested or created, try to infer the best chart
            if len(visualizations) == 1:  # Only table exists
                try:
                    visualizations.update(self._infer_best_visualizations(df))
                except Exception as inference_error:
                    print(f"⚠️ Visualization inference failed: {inference_error}")
                    visualizations["inference_error"] = f"Visualization inference failed: {str(inference_error)}"
            
            return visualizations
            
        except Exception as e:
            print(f"❌ Critical visualization error: {str(e)}")
            return {"error": f"Visualization generation failed: {str(e)}", "table": results[:20] if results else []}
    
    def _infer_best_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infer the best visualizations based on data characteristics.
        """
        visualizations = {}
        
        # If we have time series data, create line chart
        if any(col.lower() in ['date', 'time', 'year', 'month'] for col in df.columns):
            visualizations["line_chart"] = self._create_line_chart(df)
        
        # If we have categorical data, create bar chart
        elif len(df.columns) >= 2:
            # Check if we have numerical and categorical columns
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) >= 1:
                visualizations["bar_chart"] = self._create_bar_chart(df)
        
        return visualizations
    
    def _create_line_chart(self, df: pd.DataFrame) -> str:
        """
        Create a line chart using Plotly.
        """
        try:
            # Find suitable columns for x and y axes
            num_cols = df.select_dtypes(include=['number']).columns
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if len(num_cols) == 0:
                return "No numerical data available for line chart"
            
            # Use date column as x-axis if available, otherwise use index
            x_col = date_cols[0] if date_cols else df.index
            y_col = num_cols[0]  # Use first numerical column
            
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Convert to HTML
            html_string = fig.to_html(include_plotlyjs=False, full_html=False)
            return html_string
            
        except Exception as e:
            return f"Line chart generation failed: {str(e)}"
    
    def _create_bar_chart(self, df: pd.DataFrame) -> str:
        """
        Create a bar chart using Plotly.
        """
        try:
            # Find suitable columns for x and y axes
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(num_cols) == 0:
                return "No numerical data available for bar chart"
            
            # Use categorical column as x-axis if available, otherwise use index
            x_col = cat_cols[0] if len(cat_cols) > 0 else df.index[:10]  # Limit to 10 items
            y_col = num_cols[0]  # Use first numerical column
            
            # Limit data for better visualization
            plot_data = df.head(20)  # Limit to 20 items
            
            fig = px.bar(plot_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Convert to HTML
            html_string = fig.to_html(include_plotlyjs=False, full_html=False)
            return html_string
            
        except Exception as e:
            return f"Bar chart generation failed: {str(e)}"
    
    def _create_pie_chart(self, df: pd.DataFrame) -> str:
        """
        Create a pie chart using Plotly.
        """
        try:
            # Find suitable columns for labels and values
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(num_cols) == 0:
                return "No numerical data available for pie chart"
            
            # Use categorical column as labels if available, otherwise use index
            labels_col = cat_cols[0] if len(cat_cols) > 0 else df.index[:10]  # Limit to 10 items
            values_col = num_cols[0]  # Use first numerical column
            
            # Limit data for better visualization
            plot_data = df.head(10)  # Limit to 10 items
            
            fig = px.pie(plot_data, values=values_col, names=labels_col, 
                        title=f"Distribution of {values_col}")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Convert to HTML
            html_string = fig.to_html(include_plotlyjs=False, full_html=False)
            return html_string
            
        except Exception as e:
            return f"Pie chart generation failed: {str(e)}"
    
    def _create_scatter_plot(self, df: pd.DataFrame) -> str:
        """
        Create a scatter plot using Plotly.
        """
        try:
            # Find suitable columns for x and y axes
            num_cols = df.select_dtypes(include=['number']).columns
            
            if len(num_cols) < 2:
                return "Need at least 2 numerical columns for scatter plot"
            
            x_col = num_cols[0]  # Use first numerical column
            y_col = num_cols[1]  # Use second numerical column
            
            # Limit data for better visualization
            plot_data = df.head(100)  # Limit to 100 points
            
            fig = px.scatter(plot_data, x=x_col, y=y_col, 
                           title=f"Correlation between {x_col} and {y_col}")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            # Convert to HTML
            html_string = fig.to_html(include_plotlyjs=False, full_html=False)
            return html_string
            
        except Exception as e:
            return f"Scatter plot generation failed: {str(e)}"

    def process_query(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Main method to process a natural language query end-to-end.
        Returns a dictionary with SQL query, results, natural language answer, and visualizations.
        """
        try:
            # Step 1: Generate SQL query
            print("🤖 Analyzing question and generating SQL query...")
            sql_query, confidence = self.generate_sql_query(question, context)
            
            if confidence < 0.3:
                print("⚠️ Low confidence in generated query, double-checking...")
            
            # Step 2: Execute query
            print("💾 Executing SQL query...")
            results, error = self.execute_query(sql_query)
            
            if error:
                # Try a simpler fallback query
                print("⚠️ Primary query failed, trying fallback query...")
                fallback_query = "SELECT COUNT(*) as total_records FROM cust_data;"
                results, fallback_error = self.execute_query(fallback_query)
                
                if fallback_error:
                    return {
                        "success": False,
                        "error": f"Database error: {error}",
                        "sql_query": sql_query,
                        "confidence": confidence
                    }
                else:
                    sql_query = fallback_query
                    confidence = 0.3  # Lower confidence for fallback
            
            # Step 3: Generate natural language answer
            print("🗣️ Generating natural language answer...")
            try:
                answer = self.generate_natural_language_answer(question, sql_query, results)
            except Exception as answer_error:
                print(f"⚠️ Natural language answer generation failed: {answer_error}")
                # Fallback answer
                answer = f"Based on the query results, there are {len(results)} records that match your question."
            
            # Step 4: Generate visualizations
            print("📊 Generating visualizations...")
            try:
                # Get suggested visualization type from analysis
                analysis = self.analyze_question(question)
                suggested_viz = analysis.get('suggested_visualization', 'table')
                visualizations = self.generate_visualizations(results, suggested_viz)
            except Exception as viz_error:
                print(f"⚠️ Visualization generation failed: {viz_error}")
                # Fallback to basic table visualization
                visualizations = {"table": results[:20] if results else []}
                analysis = {}
            
            return {
                "success": True,
                "question": question,
                "sql_query": sql_query,
                "confidence": confidence,
                "results": results,
                "answer": answer,
                "visualizations": visualizations,
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"❌ Critical error in query processing: {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "question": question,
                "sql_query": "-- No query generated due to error",
                "confidence": 0.0,
                "results": [],
                "answer": "Sorry, I encountered an error while processing your question. Please try rephrasing or ask a different question.",
                "visualizations": {"error": "Visualization generation failed"},
                "analysis": {}
            }
