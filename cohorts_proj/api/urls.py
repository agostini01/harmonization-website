from django.urls import path

from .views import DatasetUploadView, GraphRequestView

urlpatterns = [
    # TODO - remove this path
    path('', DatasetUploadView.as_view(), name='welcome-api'),

    path('dataset-upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('get-plot/', GraphRequestView.as_view(), name='get-plot'),
]
