from rest_framework import generics

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

class DatasetUploadView(generics.CreateAPIView):
    """Handles only POST methods."""
    serializer_class = DatasetUploadSerializer
    queryset = DatasetUploadModel.objects.all()
