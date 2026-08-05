from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('label/', views.active_labeling, name='active_labeling'),
]