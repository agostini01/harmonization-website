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


class GraphRequestSerializer(serializers.Serializer):
    """To process plot requests."""

    plot_type = serializers.CharField()
    x_feature = serializers.CharField()
    y_feature = serializers.CharField()
    color_by = serializers.CharField()
    fig_dpi = serializers.IntegerField()
    plot_name = serializers.CharField()
