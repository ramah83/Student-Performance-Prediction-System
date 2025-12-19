from django.urls import path
from . import views

urlpatterns = [
    path('perform/', views.perform_clustering, name='perform_clustering'),
    path('visualization/', views.get_cluster_visualization, name='cluster_visualization'),
    path('dashboard/', views.clustering_dashboard, name='clustering_dashboard'),
]