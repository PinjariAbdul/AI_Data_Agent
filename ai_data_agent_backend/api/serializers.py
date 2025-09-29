from rest_framework import serializers
from .models import *

class QueryRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)
    query_context = serializers.CharField(max_length=2000, required=False, allow_blank=True)

class QueryResponseSerializer(serializers.Serializer):
    success = serializers.BooleanField()
    question = serializers.CharField()
    sql_query = serializers.CharField()
    confidence = serializers.FloatField()
    results = serializers.JSONField()
    answer = serializers.CharField()
    visualizations = serializers.JSONField(required=False)
    analysis = serializers.JSONField(required=False)
    error = serializers.CharField(required=False)

class DatabaseSchemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatabaseSchema
        fields = '__all__'

class SchemaAnalysisSerializer(serializers.Serializer):
    table_name = serializers.CharField()
    columns = serializers.JSONField()
    sample_data = serializers.JSONField()
    data_quality_issues = serializers.JSONField()
    suggested_meanings = serializers.JSONField()