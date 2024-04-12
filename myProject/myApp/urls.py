from . import views
from django.urls import path

urlpatterns = [
    path('api/upload_image', views.upload_image, name='upload_image'),
]
