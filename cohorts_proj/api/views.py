from rest_framework import generics, views
from django.http import HttpResponse

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

from .serializers import GraphRequestSerializer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
iris = pd.read_csv("staticfiles/datasets/iris.csv", names=col)
iris_setosa = iris.loc[iris["type"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["type"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["type"] == "Iris-versicolor"]


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


class GraphRequestView(views.APIView):
    """Handles only POST methods."""
    serializer_class = GraphRequestSerializer
    # queryset = ()

    """
    Concrete view for listing a queryset or creating a model instance.
    """

    def get(self, request, *args, **kwargs):
        return self.getPlot(request)

    @classmethod
    def getPlot(cls, request):
        """Called during get request to generate plots."""

        plot_type = request.data['plot_type']
        x_feature = request.data['x_feature']
        y_feature = request.data['y_feature']
        color_by = request.data['color_by']
        fig_dpi = int(request.data['fig_dpi'])

        t = plot_type
        gr = None

        plt.clf()
        if (t == 'scatter_plot'):
            gr = cls.getScatterPlot(x_feature, y_feature, color_by)

        if (t == 'pair_plot'):
            gr = cls.getPairPlot(x_feature, y_feature, color_by)

        if (t == 'cat_plot'):
            gr = cls.getCatPlot(x_feature, y_feature, color_by)

        if (t == 'violin_cat_plot'):
            gr = cls.getViolinCatPlot(x_feature, y_feature, color_by)

        # TODO: Add histogram

        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg", dpi=fig_dpi)

        return response

    @classmethod
    def getScatterPlot(cls, x_feature, y_feature, color_by):
        gr = sns.scatterplot(
            data=iris, x=x_feature, y=y_feature, hue=color_by)

        return gr.figure

    @classmethod
    def getPairPlot(cls, x_feature, y_feature, color_by):
        gr = sns.pairplot(
            iris, vars=[x_feature, y_feature], hue=color_by, height=3)

        return gr

    @classmethod
    def getCatPlot(cls, x_feature, y_feature, color_by):
        gr = sns.catplot(data=iris, x=x_feature,
                         y=y_feature, hue=color_by)

        return gr

    @classmethod
    def getViolinCatPlot(cls, x_feature, y_feature, color_by):
        gr = sns.catplot(data=iris, x=x_feature,
                         y=y_feature, hue=color_by, kind="violin")

        return gr
