from rest_framework import generics, views
from django.http import HttpResponse

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

from .serializers import GraphRequestSerializer

from datasets.models import RawFlower, RawUNM, RawNEU, RawDAR

from api import adapters
from api import graphs

import matplotlib.pyplot as plt
import pandas as pd


def saveFlowersToDB(csv_file):
    df = pd.read_csv(csv_file,
                     skip_blank_lines=True,
                     header=0)
    df['PIN_ID'] = range(len(df))

    # Delete database
    RawFlower.objects.all().delete()

    for entry in df.itertuples():
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


def saveUNMToDB(csv_file):
    df = pd.read_csv(csv_file,
                     skip_blank_lines=True,
                     header=0)

    # TODO droping if no outcome was provided
    df.dropna(subset=['PretermBirth'], inplace=True)
    df['PretermBirth'] = df['PretermBirth'].astype(int)
    df['Member_c'] = df['Member_c'].astype(int)
    df['TimePeriod'] = df['TimePeriod'].astype(int)

    # Delete database
    RawUNM.objects.all().delete()

    for entry in df.itertuples():
        entry = RawUNM.objects.create(
            PIN_Patient=entry.PIN_Patient,
            Member_c=entry.Member_c,
            TimePeriod=entry.TimePeriod,
            Analyte=entry.Analyte,
            Result=entry.Result,
            Creat_Corr_Result=entry.Creat_Corr_Result,
            Outcome=entry.PretermBirth,
        )


def saveNEUToDB(csv_file):
    df = pd.read_csv(csv_file,
                     skip_blank_lines=True,
                     header=0)

    # TODO droping if no outcome was provided
    df.dropna(subset=['PretermBirth'], inplace=True)
    df['PretermBirth'] = df['PretermBirth'].astype(int)
    df['Member_c'] = df['Member_c'].astype(int)
    df['TimePeriod'] = df['TimePeriod'].astype(int)

    # Delete database
    RawNEU.objects.all().delete()

    for entry in df.itertuples():
        entry = RawNEU.objects.create(
            PIN_Patient=entry.PIN_Patient,
            Member_c=entry.Member_c,
            TimePeriod=entry.TimePeriod,
            LOD=entry.LOD,
            Analyte=entry.Analyte,
            Result=entry.Result,
            Outcome=entry.PretermBirth,
        )


def saveDARToDB(csv_file):
    df = pd.read_csv(csv_file,
                     skip_blank_lines=True,
                     header=0)

    df.dropna(subset=['preterm'], inplace=True)

    # Delete database
    RawDAR.objects.all().delete()

    for entry in df.itertuples():
        entry = RawDAR.objects.create(
            unq_id=entry.unq_id,
            assay=entry.assay,
            # lab=entry.lab,
            participant_type=entry.participant_type,
            time_period=entry.time_period,
            batch=entry.batch,
            # squid=entry.squid,
            preterm=entry.preterm,
            sample_gestage_days=entry.sample_gestage_days,
            urine_specific_gravity=entry.urine_specific_gravity,
            iAs=entry.iAs, iAs_IDL=entry.iAs_IDL, iAs_BDL=entry.iAs_BDL,
            AsB=entry.AsB, AsB_IDL=entry.AsB_IDL, AsB_BDL=entry.AsB_BDL,
            DMA=entry.DMA, DMA_IDL=entry.DMA_IDL, DMA_BDL=entry.DMA_BDL,
            MMA=entry.MMA, MMA_IDL=entry.MMA_IDL, MMA_BDL=entry.MMA_BDL,
            Ba=entry.Ba, Ba_IDL=entry.Ba_IDL, Ba_BDL=entry.Ba_BDL,
            Cs=entry.Cs, Cs_IDL=entry.Cs_IDL, Cs_BDL=entry.Cs_BDL,
            Sr=entry.Sr, Sr_IDL=entry.Sr_IDL, Sr_BDL=entry.Sr_BDL,
            Ag=entry.Ag, Ag_IDL=entry.Ag_IDL, Ag_BDL=entry.Ag_BDL,
            Al=entry.Al, Al_IDL=entry.Al_IDL, Al_BDL=entry.Al_BDL,
            As=entry.As, As_IDL=entry.As_IDL, As_BDL=entry.As_BDL,
            Be=entry.Be, Be_IDL=entry.Be_IDL, Be_BDL=entry.Be_BDL,
            Cd=entry.Cd, Cd_IDL=entry.Cd_IDL, Cd_BDL=entry.Cd_BDL,
            Co=entry.Co, Co_IDL=entry.Co_IDL, Co_BDL=entry.Co_BDL,
            Cr=entry.Cr, Cr_IDL=entry.Cr_IDL, Cr_BDL=entry.Cr_BDL,
            Cu=entry.Cu, Cu_IDL=entry.Cu_IDL, Cu_BDL=entry.Cu_BDL,
            Fe=entry.Fe, Fe_IDL=entry.Fe_IDL, Fe_BDL=entry.Fe_BDL,
            Hg=entry.Hg, Hg_IDL=entry.Hg_IDL, Hg_BDL=entry.Hg_BDL,
            Mn=entry.Mn, Mn_IDL=entry.Mn_IDL, Mn_BDL=entry.Mn_BDL,
            Mo=entry.Mo, Mo_IDL=entry.Mo_IDL, Mo_BDL=entry.Mo_BDL,
            Ni=entry.Ni, Ni_IDL=entry.Ni_IDL, Ni_BDL=entry.Ni_BDL,
            Pb=entry.Pb, Pb_IDL=entry.Pb_IDL, Pb_BDL=entry.Pb_BDL,
            Sb=entry.Sb, Sb_IDL=entry.Sb_IDL, Sb_BDL=entry.Sb_BDL,
            Se=entry.Se, Se_IDL=entry.Se_IDL, Se_BDL=entry.Se_BDL,
            Sn=entry.Sn, Sn_IDL=entry.Sn_IDL, Sn_BDL=entry.Sn_BDL,
            Tl=entry.Tl, Tl_IDL=entry.Tl_IDL, Tl_BDL=entry.Tl_BDL,
            U=entry.U, U_IDL=entry.U_IDL, U_BDL=entry.U_BDL,
            W=entry.W, W_IDL=entry.W_IDL, W_BDL=entry.W_BDL,
            Zn=entry.Zn, Zn_IDL=entry.Zn_IDL, Zn_BDL=entry.Zn_BDL,
            V=entry.V, V_IDL=entry.V_IDL, V_BDL=entry.V_BDL,
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

        if request.data['dataset_type'] == 'UNM_dataset':
            csv_file = request.data['dataset_file']
            saveUNMToDB(csv_file)

        if request.data['dataset_type'] == 'NEU_dataset':
            csv_file = request.data['dataset_file']
            saveNEUToDB(csv_file)

        if request.data['dataset_type'] == 'Dartmouth_dataset':
            csv_file = request.data['dataset_file']
            saveDARToDB(csv_file)

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
        time_period = int(request.data['time_period'])
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
            df['CohortType'] = 'Flowers'

        if dataset_type == 'unm_dataset':
            df = adapters.unm.get_dataframe()

        if dataset_type == 'neu_dataset':
            df = adapters.neu.get_dataframe()

        if dataset_type == 'dar_dataset':
            df = adapters.dar.get_dataframe()

        if dataset_type == 'unmneu_dataset':
            df1 = adapters.unm.get_dataframe()
            df2 = adapters.neu.get_dataframe()
            df = pd.concat([df1, df2])

        if dataset_type == 'neudar_dataset':
            df1 = adapters.neu.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            df = pd.concat([df1, df2])

        if dataset_type == 'darunm_dataset':
            df1 = adapters.unm.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            df = pd.concat([df1, df2])

        # This is the harmonized dataset
        if dataset_type == 'har_dataset':
            # TODO Handle early exit when selected columns are not present
            # selected_columns = [x_feature, y_feature, color_by, 'CohortType']
            # TODO add NEU dataset here
            df1 = adapters.unm.get_dataframe()  # [selected_columns]
            df2 = adapters.neu.get_dataframe()  # [selected_columns]
            df3 = adapters.dar.get_dataframe()  # [selected_columns]
            df = pd.concat([df1, df2, df3])

        # Apply Filters
        if time_period != 9:
            df = df[df['TimePeriod'] == time_period]
        else:
            pass

        # Build response figure
        response = HttpResponse(content_type="image/jpg")
        plt.clf()

        # Is there data after the filters?
        if(df.shape[0] == 0):
            print('No data to plot after the filters')
            gr = graphs.noDataMessage()
            gr.savefig(response, format="jpg", dpi=fig_dpi)

        else:
            # Plot Graphs
            if (t == 'scatter_plot'):
                gr = cls.getScatterPlot(df, x_feature, y_feature, color_by)

            if (t == 'individual_scatter_plot'):
                gr = cls.getIndividualScatterPlot(
                    df, x_feature, y_feature, color_by)

            if (t == 'pair_plot'):
                gr = cls.getPairPlot(df, x_feature, y_feature, color_by)

            if (t == 'cat_plot'):
                gr = cls.getCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'violin_cat_plot'):
                gr = cls.getViolinCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'histogram_plot'):
                gr = cls.getHistogramPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_plot'):
                gr = cls.getRegPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_with_color_plot'):
                gr = cls.getRegColorPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_detailed_plot'):
                gr = cls.getRegDetailedPlot(df, x_feature, y_feature, color_by)

        # response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg", dpi=fig_dpi)

        return response

    @classmethod
    def getIndividualScatterPlot(cls, data, x_feature, y_feature, color_by,
                                 info=True):
        if info:
            return graphs.getIndividualScatterPlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getIndividualScatterPlotWithInfo(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getScatterPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getScatterPlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getScatterPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getPairPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getPairPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getPairPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getCatPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getCatPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getCatPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getViolinCatPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getViolinCatPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getViolinCatPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getHistogramPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getHistogramPlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getHistogramPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getRegPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getRegPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getRegPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getRegColorPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getRegColorPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getRegColorPlot(
                data, x_feature, y_feature, color_by)

    @classmethod
    def getRegDetailedPlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getRegDetailedPlot(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getRegDetailedPlot(
                data, x_feature, y_feature, color_by)
