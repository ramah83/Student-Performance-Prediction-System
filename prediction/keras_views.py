from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .keras_models import KerasPredictor
import os


try:
    keras_predictor = KerasPredictor()
except ImportError:
    keras_predictor = None

@api_view(['POST'])
def train_keras_model(request):
    """Train Keras neural network models and save as H5 files"""
    try:
        global keras_predictor
        
        if keras_predictor is None:
            return Response({
                'error': 'TensorFlow غير متاح - يرجى تثبيته أولاً',
                'message': 'استخدم النماذج التقليدية بدلاً من ذلك'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        
        scores = keras_predictor.train_models()
        
        return Response({
            'message': 'Keras models trained successfully and saved as H5 files',
            'scores': scores,
            'model_location': keras_predictor.model_dir
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def predict_keras_scores(request):
    """Predict student scores using Keras H5 models"""
    try:
        global keras_predictor
        
        
        if not keras_predictor.models:
            success = keras_predictor.load_models()
            if not success:
                return Response({
                    'error': 'Keras models not found. Please train the models first.'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        
        data = request.data
        
        
        predictions = keras_predictor.predict(data)
        
        return Response({
            'predictions': predictions,
            'input_data': data,
            'model_type': 'Keras Neural Network (H5)'
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def list_h5_models(request):
    """List all available H5 model files"""
    try:
        model_dir = 'models/keras/'
        
        if not os.path.exists(model_dir):
            return Response({
                'models': [],
                'message': 'No Keras models directory found'
            })
        
        h5_files = []
        for file in os.listdir(model_dir):
            if file.endswith('.h5'):
                file_path = os.path.join(model_dir, file)
                file_size = os.path.getsize(file_path)
                h5_files.append({
                    'filename': file,
                    'path': file_path,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        
        return Response({
            'models': h5_files,
            'total_models': len(h5_files),
            'model_directory': model_dir
        })
    
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)