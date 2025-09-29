from django.core.management.base import BaseCommand
from api.ai_agent import SQLQueryAgent
from django.db import connection
import json

class Command(BaseCommand):
    help = 'Demonstrate AI Data Agent capabilities with sample queries'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--question',
            type=str,
            help='Custom question to ask the AI agent',
        )
    
    def handle(self, *args, **options):
        self.stdout.write('🤖 AI Data Agent Demo')
        self.stdout.write('=' * 50)
        
        # Check if we have data
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM cust_data")
            customer_count = cursor.fetchone()[0]
            
        if customer_count == 0:
            self.stdout.write(
                self.style.ERROR('No sample data found. Run: python manage.py populate_sample_data')
            )
            return
        
        self.stdout.write(f'📊 Database ready with {customer_count} customers')
        
        # Initialize AI agent
        try:
            agent = SQLQueryAgent()
            self.stdout.write('✅ AI Agent initialized successfully')
        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'⚠️  AI Agent initialization issue: {str(e)}')
            )
            self.stdout.write('💡 Note: AI features require OpenAI API key in .env file')
            agent = None
        
        if options['question']:
            # Process custom question
            if agent:
                self.process_question(agent, options['question'])
            else:
                self.stdout.write('❌ Cannot process AI question without proper configuration')
        else:
            # Run demo queries
            self.run_demo_queries(agent)
    
    def process_question(self, agent, question):
        """Process a single question"""
        self.stdout.write(f'\n🔍 Processing: "{question}"')
        self.stdout.write('-' * 40)
        
        try:
            result = agent.process_question(question)
            
            self.stdout.write(f'🔧 Generated SQL: {result["sql_query"]}')
            self.stdout.write(f'📊 Confidence: {result["confidence_score"]:.2f}')
            self.stdout.write(f'📝 Rows returned: {len(result["result_data"])}')
            self.stdout.write(f'💬 Answer: {result["natural_language_answer"]}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
    
    def run_demo_queries(self, agent):
        """Run demonstration queries"""
        demo_questions = [
            "How many customers do we have?",
            "What are the top 3 products by price?",
            "Show me customers from New York",
            "How many orders were placed last month?",
            "What's the average salary by department?",
        ]
        
        self.stdout.write('\n🎯 Demo Questions:')
        
        for i, question in enumerate(demo_questions, 1):
            self.stdout.write(f'\n{i}. {question}')
            
            if agent:
                try:
                    result = agent.process_question(question)
                    self.stdout.write(f'   🔧 SQL: {result["sql_query"][:60]}...')
                    self.stdout.write(f'   📊 Confidence: {result["confidence_score"]:.2f}')
                    self.stdout.write(f'   📝 Data rows: {len(result["result_data"])}')
                except Exception as e:
                    self.stdout.write(f'   ❌ Error: {str(e)}')
            else:
                # Show manual SQL for demo purposes
                manual_queries = {
                    "How many customers do we have?": "SELECT COUNT(*) FROM cust_data",
                    "What are the top 3 products by price?": "SELECT col1, col3 FROM prod_tbl ORDER BY col3 DESC LIMIT 3",
                    "Show me customers from New York": "SELECT nm, cty FROM cust_data WHERE cty = 'New York'",
                    "How many orders were placed last month?": "SELECT COUNT(*) FROM Order_History WHERE order_date_time >= date('now', '-1 month')",
                    "What's the average salary by department?": "SELECT department, AVG(salary) FROM EMP_RECORDS GROUP BY department"
                }
                
                if question in manual_queries:
                    sql = manual_queries[question]
                    self.stdout.write(f'   🔧 Manual SQL: {sql}')
                    
                    try:
                        with connection.cursor() as cursor:
                            cursor.execute(sql)
                            results = cursor.fetchall()
                            self.stdout.write(f'   📝 Results: {len(results)} rows')
                    except Exception as e:
                        self.stdout.write(f'   ❌ SQL Error: {str(e)}')
        
        self.stdout.write('\n🎉 Demo completed!')
        self.stdout.write('\n💡 Tips:')
        self.stdout.write('   - Add OpenAI API key to .env for full AI features')
        self.stdout.write('   - Use --question "your question" for custom queries')
        self.stdout.write('   - API endpoints available at http://127.0.0.1:8002/api/')
        self.stdout.write('   - Test with: python api_tester.py')