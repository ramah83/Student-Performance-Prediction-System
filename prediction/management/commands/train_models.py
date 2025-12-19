from django.core.management.base import BaseCommand
from django.http import HttpRequest
from prediction.views import train_model
from rest_framework.test import APIRequestFactory
import json

class Command(BaseCommand):
    help = 'Train prediction models from command line'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting model training...'))
        
        try:
            
            factory = APIRequestFactory()
            request = factory.post('/api/prediction/train/')
            
            
            response = train_model(request)
            
            if response.status_code == 200:
                data = response.data
                self.stdout.write(self.style.SUCCESS('✅ Models trained successfully!'))
                self.stdout.write(f"Scores: {data.get('scores', {})}")
                
                if 'data_info' in data:
                    info = data['data_info']
                    self.stdout.write(f"Total records: {info.get('total_records', 'N/A')}")
                    self.stdout.write(f"Features used: {len(info.get('features_used', []))}")
                
            else:
                self.stdout.write(self.style.ERROR(f'❌ Training failed: {response.data}'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error during training: {str(e)}'))