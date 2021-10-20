from django.urls import path

from .views import DatasetUploadView, GraphRequestView, InfoRequestView

urlpatterns = [
    # TODO - remove this path
    path('', DatasetUploadView.as_view(), name='welcome-api'),

    path('dataset-upload/', DatasetUploadView.as_view(), name='dataset-upload'),
    path('get-plot/', GraphRequestView.as_view(), name='get-plot'),
    path('get-info/', InfoRequestView.as_view(), name='get-info'),
    path('get-overview/', GraphRequestView.as_view(), name='get-info'),


]
