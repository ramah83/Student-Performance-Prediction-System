from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_dashboard, name='main_dashboard'),
    path('api/data/', views.get_dashboard_data, name='dashboard_data'),
]