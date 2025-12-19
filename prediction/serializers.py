from rest_framework import serializers
from .models import StudentData, PredictionResult

class StudentDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentData
        fields = '__all__'

class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = '__all__'