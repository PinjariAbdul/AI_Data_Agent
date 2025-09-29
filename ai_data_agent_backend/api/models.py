from django.db import models
from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
import json

# Create your models here.

# Customer table with bad schema (poorly named fields, inconsistent data)
class cust_data(models.Model):  # Intentionally bad table name
    id = models.AutoField(primary_key=True)
    nm = models.CharField(max_length=255, null=True, blank=True)  # name
    emladdr = models.EmailField(null=True, blank=True)  # email address
    ph_num = models.CharField(max_length=20, null=True, blank=True)  # phone number
    addr_ln1 = models.CharField(max_length=255, null=True, blank=True)  # address line 1
    addr_ln2 = models.CharField(max_length=255, null=True, blank=True)  # address line 2
    cty = models.CharField(max_length=100, null=True, blank=True)  # city
    st = models.CharField(max_length=50, null=True, blank=True)  # state
    zip_cd = models.CharField(max_length=10, null=True, blank=True)  # zip code
    cntry = models.CharField(max_length=50, null=True, blank=True)  # country
    reg_dt = models.DateTimeField(null=True, blank=True)  # registration date
    lst_login = models.DateTimeField(null=True, blank=True)  # last login
    is_actv = models.BooleanField(default=True)  # is active
    cust_type = models.CharField(max_length=20, null=True, blank=True)  # customer type
    
    class Meta:
        db_table = 'cust_data'

# Product table with unnamed/unclear columns
class prod_tbl(models.Model):  # Intentionally bad table name
    id = models.AutoField(primary_key=True)
    col1 = models.CharField(max_length=255, null=True, blank=True)  # product name
    col2 = models.TextField(null=True, blank=True)  # description
    col3 = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  # price
    col4 = models.CharField(max_length=100, null=True, blank=True)  # category
    col5 = models.CharField(max_length=50, null=True, blank=True)  # brand
    col6 = models.IntegerField(null=True, blank=True)  # stock quantity
    col7 = models.CharField(max_length=20, null=True, blank=True)  # SKU
    col8 = models.BooleanField(default=True)  # is available
    col9 = models.DateTimeField(null=True, blank=True)  # created date
    col10 = models.CharField(max_length=255, null=True, blank=True)  # supplier info
    
    class Meta:
        db_table = 'prod_tbl'

# Orders table with mixed naming conventions
class Order_History(models.Model):  # Mixed case table name
    order_ID = models.AutoField(primary_key=True)  # Mixed case field
    customer_ref = models.ForeignKey(cust_data, on_delete=models.CASCADE, null=True, blank=True)
    order_date_time = models.DateTimeField(null=True, blank=True)
    total_amt = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    order_status = models.CharField(max_length=50, null=True, blank=True)
    payment_method = models.CharField(max_length=50, null=True, blank=True)
    shipping_addr = models.TextField(null=True, blank=True)
    delivery_date = models.DateTimeField(null=True, blank=True)
    discount_applied = models.DecimalField(max_digits=8, decimal_places=2, default=0.00)
    tax_amount = models.DecimalField(max_digits=8, decimal_places=2, default=0.00)
    
    class Meta:
        db_table = 'Order_History'

# Order items with inconsistent field naming
class orderitems(models.Model):  # lowercase table name
    item_id = models.AutoField(primary_key=True)
    ORDER_REF = models.ForeignKey(Order_History, on_delete=models.CASCADE)  # UPPERCASE field
    product_ref = models.ForeignKey(prod_tbl, on_delete=models.CASCADE, null=True, blank=True)
    qty = models.IntegerField(null=True, blank=True)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    total_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    item_discount = models.DecimalField(max_digits=8, decimal_places=2, default=0.00)
    
    class Meta:
        db_table = 'orderitems'

# Sales data with completely unclear column names
class sales_data_2023(models.Model):  # Year in table name
    id = models.AutoField(primary_key=True)
    a1 = models.CharField(max_length=100, null=True, blank=True)  # region
    b2 = models.CharField(max_length=100, null=True, blank=True)  # sales rep
    c3 = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)  # revenue
    d4 = models.IntegerField(null=True, blank=True)  # units sold
    e5 = models.DateField(null=True, blank=True)  # sale date
    f6 = models.CharField(max_length=50, null=True, blank=True)  # product category
    g7 = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True)  # commission
    h8 = models.CharField(max_length=20, null=True, blank=True)  # quarter
    
    class Meta:
        db_table = 'sales_data_2023'

# Employee table with mixed data quality
class EMP_RECORDS(models.Model):  # ALL CAPS table name
    emp_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=100, null=True, blank=True)
    last_name = models.CharField(max_length=100, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    department = models.CharField(max_length=100, null=True, blank=True)
    position = models.CharField(max_length=100, null=True, blank=True)
    salary = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    hire_date = models.DateField(null=True, blank=True)
    manager_id = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'EMP_RECORDS'

# Database schema metadata to help the AI agent understand the structure
class DatabaseSchema(models.Model):
    table_name = models.CharField(max_length=255)
    column_name = models.CharField(max_length=255)
    actual_meaning = models.CharField(max_length=255)  # What the column actually represents
    data_type = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    sample_values = models.TextField(null=True, blank=True)  # JSON string of sample values
    data_quality_issues = models.TextField(null=True, blank=True)
    
    class Meta:
        db_table = 'database_schema'
        unique_together = ['table_name', 'column_name']

# Model for tracking uploaded Excel files
class UploadedFile(models.Model):
    file_name = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.BigIntegerField()
    processed = models.BooleanField(default=False)
    table_name = models.CharField(max_length=255, null=True, blank=True)  # Generated table name
    sheet_count = models.IntegerField(default=0)
    row_count = models.IntegerField(default=0)
    column_count = models.IntegerField(default=0)
    processing_errors = models.TextField(null=True, blank=True)
    data_quality_summary = models.TextField(null=True, blank=True)  # JSON string
    
    class Meta:
        db_table = 'uploaded_files'
    
    def __str__(self):
        return f"{self.original_filename} ({self.uploaded_at})"

# Model for tracking sheets within uploaded files
class FileSheet(models.Model):
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='sheets')
    sheet_name = models.CharField(max_length=255)
    original_sheet_name = models.CharField(max_length=255)
    table_name = models.CharField(max_length=255)  # Generated table name for this sheet
    row_count = models.IntegerField(default=0)
    column_count = models.IntegerField(default=0)
    has_header = models.BooleanField(default=True)
    header_row = models.IntegerField(default=0)
    data_start_row = models.IntegerField(default=1)
    processing_notes = models.TextField(null=True, blank=True)
    
    class Meta:
        db_table = 'file_sheets'
