from rest_framework import generics

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel


class DatasetUploadView(generics.CreateAPIView):
    """Handles only POST methods."""
    serializer_class = DatasetUploadSerializer
    queryset = DatasetUploadModel.objects.all()

    def post(self, request, *args, **kwargs):
        """Custom post to print exceptions during upload."""

        try:
            return self.create(request, *args, **kwargs)
        except Exception as e:
            print(e)
