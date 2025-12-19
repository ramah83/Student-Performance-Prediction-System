from django.contrib import admin
from .models import ClusterResult, StudentCluster

@admin.register(ClusterResult)
class ClusterResultAdmin(admin.ModelAdmin):
    list_display = ['cluster_id', 'cluster_name', 'student_count', 'avg_math_score', 'avg_reading_score', 'avg_writing_score', 'created_at']
    list_filter = ['created_at']

@admin.register(StudentCluster)
class StudentClusterAdmin(admin.ModelAdmin):
    list_display = ['student_id', 'cluster_result', 'distance_to_center', 'created_at']
    list_filter = ['cluster_result', 'created_at']