from rest_framework import serializers
from .models import DatasetUploadModel


class DatasetUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetUploadModel
        fields = (
            'uploader_name',
            'uploader_email',
            # 'dataset_time',
            'dataset_type',
            'dataset_file',
        )
