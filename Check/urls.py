from django.urls import path
from .import views
from .views import visualization

urlpatterns = [
    path('display/', views.visualization, name='visualization'),
    path('', views.import_model, name='model'),
]