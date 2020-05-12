from django.urls import path

from .views import UploadPageView, UploadSuccessPageView


urlpatterns = [
    path('', UploadPageView.as_view(), name='upload'),
    path('success/', UploadSuccessPageView.as_view(), name='upload-success'),
]
