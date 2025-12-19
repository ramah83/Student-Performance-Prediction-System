from django.urls import path
from . import views


try:
    from . import keras_views
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️ Keras views not available - TensorFlow not installed")

urlpatterns = [
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.predict_scores, name='predict_scores'),
    path('performance/', views.get_model_performance, name='get_model_performance'),
    path('download-report/', views.download_performance_report, name='download_performance_report'),
    path('dashboard/', views.prediction_dashboard, name='prediction_dashboard'),
    path('status/', views.check_model_status, name='check_model_status'),
]


if KERAS_AVAILABLE:
    urlpatterns += [
        path('keras/train/', keras_views.train_keras_model, name='train_keras_model'),
        path('keras/predict/', keras_views.predict_keras_scores, name='predict_keras_scores'),
        path('keras/models/', keras_views.list_h5_models, name='list_h5_models'),
    ]