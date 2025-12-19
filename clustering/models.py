from django.db import models

class ClusterResult(models.Model):
    cluster_id = models.IntegerField()
    cluster_name = models.CharField(max_length=100)
    description = models.TextField()
    student_count = models.IntegerField()
    avg_math_score = models.FloatField()
    avg_reading_score = models.FloatField()
    avg_writing_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Cluster {self.cluster_id}: {self.cluster_name}"

class StudentCluster(models.Model):
    student_id = models.IntegerField()
    cluster_result = models.ForeignKey(ClusterResult, on_delete=models.CASCADE)
    distance_to_center = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Student {self.student_id} in {self.cluster_result.cluster_name}"