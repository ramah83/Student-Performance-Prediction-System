from django.contrib import admin
from .models import StudentData, PredictionResult

@admin.register(StudentData)
class StudentDataAdmin(admin.ModelAdmin):
    list_display = ['id', 'gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation', 'math_score', 'reading_score', 'writing_score', 'created_at']
    list_filter = ['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_preparation']
    search_fields = ['id']
    ordering = ['-created_at']

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'student_data', 'predicted_math_score', 'predicted_reading_score', 'predicted_writing_score', 'model_used', 'accuracy_score', 'created_at']
    list_filter = ['model_used', 'created_at']
    search_fields = ['student_data__id']
    ordering = ['-created_at']