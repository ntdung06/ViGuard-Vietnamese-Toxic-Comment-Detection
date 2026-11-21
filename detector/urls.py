from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/health/', views.health, name='health'),
    path('api/predict/', views.predict, name='predict'),
    path('api/predict-batch/', views.predict_batch, name='predict_batch'),
    path('api/history/', views.history, name='history'),
]
