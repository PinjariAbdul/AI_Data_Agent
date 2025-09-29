"""
Smart Offline AI - No API Required
A rule-based system that generates SQL queries from natural language without external APIs
"""
import re
import json
from typing import Dict, Any, List

class OfflineAIAgent:
    """
    Offline AI that generates SQL using pattern matching and rules.
    No external API required - completely free!
    """
    
    def __init__(self):
        self.table_mapping = {
            "customer": "cust_data",
            "customers": "cust_data",
            "client": "cust_data",
            "user": "cust_data",
            "product": "prod_tbl", 
            "products": "prod_tbl",
            "item": "prod_tbl",
            "order": "Order_History",
            "orders": "Order_History",
            "purchase": "Order_History",
            "sales": "sales_data_2023",
            "revenue": "sales_data_2023",
            "employee": "EMP_RECORDS",
            "employees": "EMP_RECORDS",
            "staff": "EMP_RECORDS",
            "worker": "EMP_RECORDS"
        }
        
        self.column_mapping = {
            "cust_data": {
                "name": "nm",
                "email": "emladdr", 
                "phone": "ph_num",
                "address": "addr_ln1",
                "city": "cty",
                "state": "st",
                "country": "cntry",
                "active": "is_actv"
            },
            "prod_tbl": {
                "name": "col1",
                "description": "col2", 
                "price": "col3",
                "category": "col4",
                "brand": "col5",
                "stock": "col6"
            },
            "sales_data_2023": {
                "region": "a1",
                "sales_rep": "b2",
                "revenue": "c3",
                "units": "d4",
                "date": "e5",
                "category": "f6"
            }
        }
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Smart analysis without external APIs"""
        question_lower = question.lower()
        
        # Detect intent
        intent = "unknown"
        if any(word in question_lower for word in ["count", "how many", "number of"]):
            intent = "count_query"
        elif any(word in question_lower for word in ["top", "best", "highest", "most"]):
            intent = "top_analysis"
        elif any(word in question_lower for word in ["trend", "over time", "by month", "by year"]):
            intent = "trend_analysis"
        elif any(word in question_lower for word in ["average", "avg", "mean"]):
            intent = "aggregation_query"
        elif any(word in question_lower for word in ["where", "in", "from"]):
            intent = "filtered_query"
        
        # Detect tables
        required_tables = []
        for keyword, table in self.table_mapping.items():
            if keyword in question_lower:
                required_tables.append(table)
        
        if not required_tables:
            required_tables = ["cust_data"]  # Default table
        
        # Suggest visualization
        visualization = "table"
        if intent == "trend_analysis":
            visualization = "line_chart"
        elif intent == "top_analysis":
            visualization = "bar_chart"  
        elif intent == "count_query":
            visualization = "pie_chart"
        
        return {
            "intent": f"Smart offline analysis: {intent}",
            "required_tables": required_tables,
            "analysis_type": intent,
            "suggested_visualization": visualization
        }
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL using smart pattern matching"""
        question_lower = question.lower()
        
        # Customer queries
        if "customer" in question_lower:
            if "count" in question_lower or "how many" in question_lower:
                if "active" in question_lower:
                    return "SELECT COUNT(*) as active_customers FROM cust_data WHERE is_actv = 1;"
                else:
                    return "SELECT COUNT(*) as total_customers FROM cust_data;"
            elif "california" in question_lower or "ca" in question_lower:
                return "SELECT nm, emladdr, cty FROM cust_data WHERE st = 'CA' AND is_actv = 1;"
            elif "email" in question_lower:
                return "SELECT nm, emladdr FROM cust_data WHERE emladdr IS NOT NULL AND is_actv = 1;"
        
        # Product queries  
        if "product" in question_lower:
            if "top" in question_lower:
                return """
                SELECT p.col1 as product_name, SUM(oi.total_price) as revenue 
                FROM prod_tbl p 
                JOIN orderitems oi ON p.id = oi.product_ref 
                GROUP BY p.id, p.col1 
                ORDER BY revenue DESC 
                LIMIT 10;
                """
            elif "count" in question_lower:
                return "SELECT COUNT(*) as total_products FROM prod_tbl WHERE col8 = 1;"
            elif "category" in question_lower:
                return "SELECT col4 as category, COUNT(*) as product_count FROM prod_tbl GROUP BY col4;"
        
        # Sales queries
        if "sales" in question_lower or "revenue" in question_lower:
            if "region" in question_lower:
                return "SELECT a1 as region, SUM(c3) as total_sales FROM sales_data_2023 GROUP BY a1 ORDER BY total_sales DESC;"
            elif "trend" in question_lower or "month" in question_lower:
                return "SELECT strftime('%Y-%m', e5) as month, SUM(c3) as monthly_sales FROM sales_data_2023 GROUP BY strftime('%Y-%m', e5) ORDER BY month;"
            elif "top" in question_lower:
                return "SELECT b2 as sales_rep, SUM(c3) as total_sales FROM sales_data_2023 GROUP BY b2 ORDER BY total_sales DESC LIMIT 10;"
        
        # Order queries
        if "order" in question_lower:
            if "average" in question_lower:
                return "SELECT AVG(total_amt) as average_order_value FROM Order_History WHERE order_status = 'completed';"
            elif "count" in question_lower:
                return "SELECT COUNT(*) as total_orders FROM Order_History;"
            elif "recent" in question_lower:
                return "SELECT order_ID, customer_ref, total_amt, order_date_time FROM Order_History ORDER BY order_date_time DESC LIMIT 10;"
        
        # Employee queries
        if "employee" in question_lower or "staff" in question_lower:
            if "department" in question_lower:
                return "SELECT department, COUNT(*) as employee_count FROM EMP_RECORDS WHERE is_active = 1 GROUP BY department;"
            elif "count" in question_lower:
                return "SELECT COUNT(*) as total_employees FROM EMP_RECORDS WHERE is_active = 1;"
        
        # Default fallback
        return "SELECT COUNT(*) as total_records FROM cust_data;"
    
    def generate_explanation(self, question: str, sql_query: str, results: List[Dict]) -> str:
        """Generate natural language explanation"""
        if not results:
            return "No data found for your query. This might be due to specific criteria in your question or data availability."
        
        result_count = len(results)
        first_result = results[0] if results else {}
        
        # Generate smart explanations based on query patterns
        question_lower = question.lower()
        
        if "count" in question_lower:
            total = first_result.get(list(first_result.keys())[0], 0) if first_result else 0
            return f"Based on the analysis, there are {total} records matching your criteria. This count provides insight into the scale of your data in this category."
        
        elif "top" in question_lower:
            return f"The analysis shows the top performers in your data. {result_count} results are displayed, ranked by their performance metrics. This helps identify the most significant contributors."
        
        elif "sales" in question_lower or "revenue" in question_lower:
            return f"The sales analysis reveals patterns across {result_count} data points. This breakdown helps understand revenue distribution and performance trends in your business."
        
        elif "product" in question_lower:
            return f"The product analysis shows {result_count} items. This data helps understand your product portfolio and can inform inventory and marketing decisions."
        
        elif "customer" in question_lower:
            return f"The customer analysis covers {result_count} records. This information provides insights into your customer base and can help with targeting and retention strategies."
        
        else:
            return f"The analysis processed {result_count} records from your database. The results provide valuable business insights that can inform decision-making and strategic planning."