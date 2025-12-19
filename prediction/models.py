from django.db import models

class StudentData(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]
    
    RACE_CHOICES = [
        ('group A', 'Group A'),
        ('group B', 'Group B'),
        ('group C', 'Group C'),
        ('group D', 'Group D'),
        ('group E', 'Group E'),
    ]
    
    EDUCATION_CHOICES = [
        ('some high school', 'Some High School'),
        ('high school', 'High School'),
        ('some college', 'Some College'),
        ('associate\'s degree', 'Associate\'s Degree'),
        ('bachelor\'s degree', 'Bachelor\'s Degree'),
        ('master\'s degree', 'Master\'s Degree'),
    ]
    
    LUNCH_CHOICES = [
        ('standard', 'Standard'),
        ('free/reduced', 'Free/Reduced'),
    ]
    
    PREP_CHOICES = [
        ('none', 'None'),
        ('completed', 'Completed'),
    ]
    
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    race_ethnicity = models.CharField(max_length=20, choices=RACE_CHOICES)
    parental_education = models.CharField(max_length=30, choices=EDUCATION_CHOICES)
    lunch = models.CharField(max_length=15, choices=LUNCH_CHOICES)
    test_preparation = models.CharField(max_length=15, choices=PREP_CHOICES)
    math_score = models.IntegerField()
    reading_score = models.IntegerField()
    writing_score = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Student {self.id} - Math: {self.math_score}"

class PredictionResult(models.Model):
    student_data = models.ForeignKey(StudentData, on_delete=models.CASCADE)
    predicted_math_score = models.FloatField(null=True, blank=True)
    predicted_reading_score = models.FloatField(null=True, blank=True)
    predicted_writing_score = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=50)
    accuracy_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for Student {self.student_data.id}"