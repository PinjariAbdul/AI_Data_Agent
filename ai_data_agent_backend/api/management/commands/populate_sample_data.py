from django.core.management.base import BaseCommand
from api.models import *
from datetime import datetime, timedelta
import random
from decimal import Decimal

class Command(BaseCommand):
    help = 'Populate database with sample data including dirty/complex data scenarios'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--clean',
            action='store_true',
            help='Clean existing data before populating',
        )
    
    def handle(self, *args, **options):
        if options['clean']:
            self.stdout.write('Cleaning existing data...')
            self.clean_data()
        
        self.stdout.write('Populating sample data...')
        
        # Populate customers with dirty data
        self.populate_customers()
        
        # Populate products with unclear column names
        self.populate_products()
        
        # Populate employees
        self.populate_employees()
        
        # Populate orders
        self.populate_orders()
        
        # Populate sales data
        self.populate_sales_data()
        
        # Populate schema metadata
        self.populate_schema_metadata()
        
        self.stdout.write(
            self.style.SUCCESS('Successfully populated database with sample data')
        )
    
    def clean_data(self):
        """Clean existing data"""
        orderitems.objects.all().delete()
        Order_History.objects.all().delete()
        sales_data_2023.objects.all().delete()
        EMP_RECORDS.objects.all().delete()
        prod_tbl.objects.all().delete()
        cust_data.objects.all().delete()
        DatabaseSchema.objects.all().delete()
    
    def populate_customers(self):
        """Populate customer data with intentional quality issues"""
        customers_data = [
            # Good data
            {
                'nm': 'John Smith',
                'emladdr': 'john.smith@email.com',
                'ph_num': '555-123-4567',
                'addr_ln1': '123 Main St',
                'cty': 'New York',
                'st': 'NY',
                'zip_cd': '10001',
                'cntry': 'USA',
                'cust_type': 'premium'
            },
            # Mixed case and formatting issues
            {
                'nm': 'MARY JOHNSON',
                'emladdr': 'mary.j@GMAIL.COM',
                'ph_num': '(555) 234-5678',
                'addr_ln1': '456 oak avenue',
                'cty': 'los angeles',
                'st': 'ca',
                'zip_cd': '90210',
                'cntry': 'usa',
                'cust_type': 'REGULAR'
            },
            # Missing data
            {
                'nm': 'Bob Wilson',
                'emladdr': None,  # Missing email
                'ph_num': '555.345.6789',
                'addr_ln1': '789 Pine St',
                'cty': None,  # Missing city
                'st': 'TX',
                'zip_cd': '75201',
                'cntry': 'USA',
                'cust_type': 'regular'
            },
            # Inconsistent formats
            {
                'nm': 'alice_cooper',
                'emladdr': 'alice@yahoo.com',
                'ph_num': '+1-555-456-7890',
                'addr_ln1': '321 Elm Street, Apt 4B',
                'cty': 'Chicago',
                'st': 'Illinois',  # Full state name instead of abbreviation
                'zip_cd': '60601-1234',  # Extended zip
                'cntry': 'United States',
                'cust_type': 'premium'
            },
            # Duplicate-like data
            {
                'nm': 'John Smith',  # Same name as first customer
                'emladdr': 'j.smith@company.com',
                'ph_num': '555-123-9999',
                'addr_ln1': '999 Different St',
                'cty': 'Boston',
                'st': 'MA',
                'zip_cd': '02101',
                'cntry': 'USA',
                'cust_type': 'regular'
            }
        ]
        
        for i, customer_data in enumerate(customers_data):
            # Add registration dates with some randomness
            reg_date = datetime.now() - timedelta(days=random.randint(30, 365))
            last_login = reg_date + timedelta(days=random.randint(1, 100))
            
            customer_data.update({
                'reg_dt': reg_date,
                'lst_login': last_login,
                'is_actv': random.choice([True, True, True, False])  # Mostly active
            })
            
            cust_data.objects.create(**customer_data)
        
        self.stdout.write('Created customers with data quality issues')
    
    def populate_products(self):
        """Populate products using unclear column names"""
        products_data = [
            {
                'col1': 'Laptop Pro 15\"',  # Product name
                'col2': 'High-performance laptop with 16GB RAM',  # Description
                'col3': Decimal('1299.99'),  # Price
                'col4': 'Electronics',  # Category
                'col5': 'TechBrand',  # Brand
                'col6': 50,  # Stock quantity
                'col7': 'LPT-15-001',  # SKU
                'col8': True,  # Is available
                'col10': 'TechSupplier Inc'  # Supplier info
            },
            {
                'col1': 'wireless mouse',  # Inconsistent capitalization
                'col2': None,  # Missing description
                'col3': Decimal('29.99'),
                'col4': 'electronics',  # Lowercase category
                'col5': 'MouseCorp',
                'col6': 0,  # Out of stock
                'col7': 'MSE-WL-002',
                'col8': False,  # Not available
                'col10': None  # Missing supplier
            },
            {
                'col1': 'Coffee Mug - Large',
                'col2': 'Ceramic coffee mug, 16oz capacity',
                'col3': Decimal('15.50'),
                'col4': 'Kitchen',
                'col5': 'HomeWare',
                'col6': 200,
                'col7': 'MUG-LG-003',
                'col8': True,
                'col10': 'Kitchen Supplies Ltd'
            },
            {
                'col1': '',  # Empty product name
                'col2': 'Mystery product with missing name',
                'col3': Decimal('99.99'),
                'col4': 'Unknown',
                'col5': '',  # Empty brand
                'col6': 5,
                'col7': 'UNK-001',
                'col8': True,
                'col10': 'Unknown Supplier'
            }
        ]
        
        for product_data in products_data:
            product_data['col9'] = datetime.now() - timedelta(days=random.randint(1, 180))
            prod_tbl.objects.create(**product_data)
        
        self.stdout.write('Created products with unclear column names')
    
    def populate_employees(self):
        """Populate employee data"""
        employees_data = [
            {
                'first_name': 'Sarah',
                'last_name': 'Johnson',
                'email': 'sarah.johnson@company.com',
                'department': 'Sales',
                'position': 'Sales Manager',
                'salary': Decimal('75000.00'),
                'hire_date': datetime(2022, 3, 15).date(),
                'manager_id': None,
                'is_active': True
            },
            {
                'first_name': 'Mike',
                'last_name': 'Chen',
                'email': 'mike.chen@company.com',
                'department': 'Sales',
                'position': 'Sales Representative',
                'salary': Decimal('55000.00'),
                'hire_date': datetime(2023, 1, 10).date(),
                'manager_id': 1,  # Reports to Sarah
                'is_active': True
            },
            {
                'first_name': 'Lisa',
                'last_name': 'Brown',
                'email': 'lisa.brown@company.com',
                'department': 'Marketing',
                'position': 'Marketing Director',
                'salary': Decimal('85000.00'),
                'hire_date': datetime(2021, 8, 20).date(),
                'manager_id': None,
                'is_active': True
            }
        ]
        
        for emp_data in employees_data:
            EMP_RECORDS.objects.create(**emp_data)
        
        self.stdout.write('Created employee records')
    
    def populate_orders(self):
        """Populate orders and order items"""
        customers = list(cust_data.objects.all())
        products = list(prod_tbl.objects.all())
        
        if not customers or not products:
            self.stdout.write('No customers or products found, skipping orders')
            return
        
        # Create orders
        for i in range(10):
            customer = random.choice(customers)
            order_date = datetime.now() - timedelta(days=random.randint(1, 90))
            
            order = Order_History.objects.create(
                customer_ref=customer,
                order_date_time=order_date,
                total_amt=Decimal('0.00'),  # Will update after adding items
                order_status=random.choice(['pending', 'completed', 'shipped', 'cancelled']),
                payment_method=random.choice(['credit_card', 'paypal', 'bank_transfer']),
                shipping_addr=f"{customer.addr_ln1}, {customer.cty}, {customer.st}",
                delivery_date=order_date + timedelta(days=random.randint(2, 10)) if random.choice([True, False]) else None,
                discount_applied=Decimal(str(random.uniform(0, 50))),
                tax_amount=Decimal('0.00')  # Will calculate
            )
            
            # Add order items
            total_amount = Decimal('0.00')
            for _ in range(random.randint(1, 4)):
                product = random.choice(products)
                quantity = random.randint(1, 5)
                unit_price = product.col3 or Decimal('10.00')  # Use product price or default
                total_price = unit_price * quantity
                
                orderitems.objects.create(
                    ORDER_REF=order,
                    product_ref=product,
                    qty=quantity,
                    unit_price=unit_price,
                    total_price=total_price,
                    item_discount=Decimal(str(random.uniform(0, 10)))
                )
                
                total_amount += total_price
            
            # Update order total
            order.total_amt = total_amount
            order.tax_amount = total_amount * Decimal('0.08')  # 8% tax
            order.save()
        
        self.stdout.write('Created orders and order items')
    
    def populate_sales_data(self):
        """Populate sales data with unclear column names"""
        regions = ['North', 'South', 'East', 'West', 'Central']
        sales_reps = ['Alice Wilson', 'Bob Smith', 'Carol Davis', 'David Lee', 'Emma Taylor']
        categories = ['Electronics', 'Kitchen', 'Office', 'Sports', 'Books']
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        
        for i in range(50):
            sales_data_2023.objects.create(
                a1=random.choice(regions),  # region
                b2=random.choice(sales_reps),  # sales rep
                c3=Decimal(str(random.uniform(1000, 50000))),  # revenue
                d4=random.randint(10, 500),  # units sold
                e5=datetime(2023, random.randint(1, 12), random.randint(1, 28)).date(),  # sale date
                f6=random.choice(categories),  # product category
                g7=Decimal(str(random.uniform(100, 5000))),  # commission
                h8=random.choice(quarters)  # quarter
            )
        
        self.stdout.write('Created sales data with unclear column names')
    
    def populate_schema_metadata(self):
        """Populate schema metadata to help the AI understand the database"""
        schema_mappings = [
            # Customer table
            ('cust_data', 'nm', 'Customer Name', 'VARCHAR', 'Customer full name'),
            ('cust_data', 'emladdr', 'Email Address', 'VARCHAR', 'Customer email address'),
            ('cust_data', 'ph_num', 'Phone Number', 'VARCHAR', 'Customer phone number'),
            ('cust_data', 'cty', 'City', 'VARCHAR', 'Customer city'),
            ('cust_data', 'st', 'State', 'VARCHAR', 'Customer state or province'),
            
            # Product table
            ('prod_tbl', 'col1', 'Product Name', 'VARCHAR', 'Name of the product'),
            ('prod_tbl', 'col2', 'Description', 'TEXT', 'Product description'),
            ('prod_tbl', 'col3', 'Price', 'DECIMAL', 'Product price in USD'),
            ('prod_tbl', 'col4', 'Category', 'VARCHAR', 'Product category'),
            ('prod_tbl', 'col5', 'Brand', 'VARCHAR', 'Product brand'),
            ('prod_tbl', 'col6', 'Stock Quantity', 'INTEGER', 'Available stock quantity'),
            ('prod_tbl', 'col7', 'SKU', 'VARCHAR', 'Stock Keeping Unit identifier'),
            ('prod_tbl', 'col8', 'Is Available', 'BOOLEAN', 'Whether product is available for sale'),
            
            # Sales data
            ('sales_data_2023', 'a1', 'Region', 'VARCHAR', 'Sales region'),
            ('sales_data_2023', 'b2', 'Sales Representative', 'VARCHAR', 'Name of sales rep'),
            ('sales_data_2023', 'c3', 'Revenue', 'DECIMAL', 'Sales revenue amount'),
            ('sales_data_2023', 'd4', 'Units Sold', 'INTEGER', 'Number of units sold'),
            ('sales_data_2023', 'e5', 'Sale Date', 'DATE', 'Date of the sale'),
            ('sales_data_2023', 'f6', 'Product Category', 'VARCHAR', 'Category of products sold'),
            ('sales_data_2023', 'g7', 'Commission', 'DECIMAL', 'Commission earned'),
            ('sales_data_2023', 'h8', 'Quarter', 'VARCHAR', 'Sales quarter (Q1, Q2, Q3, Q4)'),
        ]
        
        for table, column, meaning, data_type, description in schema_mappings:
            DatabaseSchema.objects.create(
                table_name=table,
                column_name=column,
                actual_meaning=meaning,
                data_type=data_type,
                description=description
            )
        
        self.stdout.write('Created schema metadata')