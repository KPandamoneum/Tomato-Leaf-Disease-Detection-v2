from django.urls import path
from . import views


urlpatterns = [
    path("", views.predict_disease, name="predict"),
    path("camera_capture", views.camera_capture, name="camera_capture")
]
