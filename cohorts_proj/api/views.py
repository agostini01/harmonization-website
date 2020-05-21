from rest_framework import generics, views
from django.http import HttpResponse

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

from .serializers import GraphRequestSerializer

from datasets.models import RawFlower

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def saveFlowersToDB(csv_file):
            df = pd.read_csv(csv_file,
                            skip_blank_lines=True,
                            header=0)
            # print(df.head())
            df['PIN_ID'] = range(len(df))
            # print(df.head())

            # Delete database
            RawFlower.objects.all().delete()

            for entry in df.itertuples():
                # print('>>>> HIT HERE 0')
                entry = RawFlower.objects.create(
                    # Just a number for the sample

                    PIN_ID=entry.PIN_ID,

                    # Result: value in cm
                    sepal_length=entry.sepal_length,
                    sepal_width=entry.sepal_width,
                    petal_length=entry.petal_length,
                    petal_width=entry.petal_width,

                    # The type of flower
                    flower_type=entry.flower_type
                )


class DatasetUploadView(generics.CreateAPIView):
    """Handles only POST methods."""
    serializer_class = DatasetUploadSerializer
    queryset = DatasetUploadModel.objects.all()

    def post(self, request, *args, **kwargs):
        """Saves CSV to DatasetModel database and populate raw databases."""

        if request.data['dataset_type'] == 'flowers_dataset':
            csv_file = request.data['dataset_file']
            saveFlowersToDB(csv_file)

        # Commit model upload of the regular dataset
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
        dataset_type = request.data['dataset_type']

        t = plot_type
        gr = None

        # Selects the datasets
        # It only query the database for the correct columns
        df = pd.DataFrame()
        if dataset_type == 'flowers_dataset':
            df = pd.DataFrame.from_records(
                RawFlower.objects.all().values(x_feature, y_feature, color_by))

        if dataset_type == 'unm_dataset':
            df = pd.DataFrame.from_records(
                RawUNM.objects.all().values(x_feature, y_feature, color_by))

        if dataset_type == 'neu_dataset':
            df = pd.DataFrame.from_records(
                RawNEU.objects.all().values(x_feature, y_feature, color_by))

        if dataset_type == 'dar_dataset':
            df = pd.DataFrame.from_records(
                RawDAR.objects.all().values(x_feature, y_feature, color_by))

        # This is the harmonized dataset
        if dataset_type == 'har_dataset':
            df = pd.DataFrame.from_records(
                RawHAR.objects.all().values(x_feature, y_feature, color_by))

        plt.clf()
        if (t == 'scatter_plot'):
            gr = cls.getScatterPlot(df, x_feature, y_feature, color_by)

        if (t == 'pair_plot'):
            gr = cls.getPairPlot(df, x_feature, y_feature, color_by)

        if (t == 'cat_plot'):
            gr = cls.getCatPlot(df, x_feature, y_feature, color_by)

        if (t == 'violin_cat_plot'):
            gr = cls.getViolinCatPlot(df, x_feature, y_feature, color_by)

        # TODO: Add histogram

        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg", dpi=fig_dpi)

        return response

    @classmethod
    def getScatterPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.scatterplot(
            data=data, x=x_feature, y=y_feature, hue=color_by)

        return gr.figure

    @classmethod
    def getPairPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.pairplot(
            data, vars=[x_feature, y_feature], hue=color_by, height=3)

        return gr

    @classmethod
    def getCatPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.catplot(data=data, x=x_feature,
                         y=y_feature, hue=color_by)

        return gr

    @classmethod
    def getViolinCatPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.catplot(data=data, x=x_feature,
                         y=y_feature, hue=color_by, kind="violin")

        return gr
