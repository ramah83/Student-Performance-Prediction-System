from django.core.management.base import BaseCommand
from prediction.models import StudentData
import random

class Command(BaseCommand):
    help = 'Load sample student data'

    def handle(self, *args, **options):
        
        StudentData.objects.all().delete()
        
        
        genders = ['male', 'female']
        races = ['group A', 'group B', 'group C', 'group D', 'group E']
        educations = [
            'some high school', 'high school', 'some college', 
            'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'
        ]
        lunches = ['standard', 'free/reduced']
        preps = ['none', 'completed']
        
        
        students = []
        for i in range(1000):
            student = StudentData(
                gender=random.choice(genders),
                race_ethnicity=random.choice(races),
                parental_education=random.choice(educations),
                lunch=random.choice(lunches),
                test_preparation=random.choice(preps),
                math_score=random.randint(0, 100),
                reading_score=random.randint(0, 100),
                writing_score=random.randint(0, 100)
            )
            students.append(student)
        
        
        StudentData.objects.bulk_create(students)
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {len(students)} sample students')
        )