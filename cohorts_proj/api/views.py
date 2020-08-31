from rest_framework import generics, views
from django.http import HttpResponse

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

from .serializers import GraphRequestSerializer

from datasets.models import RawFlower, RawUNM, RawDAR

from api import adapters

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats


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
            df = pd.DataFrame.from_records(
                RawNEU.objects.all().values(x_feature, y_feature, color_by))

        if dataset_type == 'dar_dataset':
            df = adapters.dar.get_dataframe()

        # This is the harmonized dataset
        if dataset_type == 'har_dataset':
            # TODO Handle early exit when selected columns are not present
            # selected_columns = [x_feature, y_feature, color_by, 'CohortType']
            df1 = adapters.unm.get_dataframe()  # [selected_columns]
            df2 = adapters.dar.get_dataframe()  # [selected_columns]
            df = pd.concat([df1, df2])

        # Apply Filters
        if time_period != 9:
            df = df[df['TimePeriod'] == time_period]
        else:
            pass

        # Is there data after the filters?
        # TODO

        # Plot Graphs
        plt.clf()
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

        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg", dpi=fig_dpi)

        return response

    @classmethod
    def getIndividualScatterPlot(cls, data, x_feature, y_feature, color_by):

        filtered_df = data[data[[x_feature, y_feature]].notnull().all(1)]
        info1 = str(filtered_df[[x_feature, y_feature]
                                ].describe(include='all'))
        info2 = str(filtered_df[[color_by]].describe(include='all'))
        info = "Summary of intersection between analytes\n" + \
            "(Not Null samples only):\n\n" + \
            info1+"\n\n"+info2

        color_by_options = data[color_by].unique()
        color_by_count = len(color_by_options)
        fig, ax = plt.subplots(1, color_by_count+1,
                               sharey=True, figsize=(5*(color_by_count+1), 5))

        for i, v in enumerate(color_by_options):
            if i > 0:
                sns.scatterplot(
                    data=data[data[color_by] == v], x=x_feature, y=y_feature,
                    hue=color_by, alpha=0.8, s=20, hue_order=color_by_options,
                    legend=False, style='CohortType', ax=ax[i])
            else:  # With legend
                sns.scatterplot(
                    data=data[data[color_by] == v], x=x_feature, y=y_feature,
                    hue=color_by, alpha=0.8, s=20, hue_order=color_by_options,
                    legend='brief', style='CohortType', ax=ax[i])

        sns.despine(ax=ax[color_by_count], left=True, bottom=True, trim=True)
        ax[color_by_count].set(xlabel=None)
        ax[color_by_count].set(xticklabels=[])

        ax[color_by_count].text(0, 0, info, style='italic',
                                bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10})

        return fig

    @classmethod
    def getScatterPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.scatterplot(
            data=data, x=x_feature, y=y_feature,
            hue=color_by, alpha=0.8, s=15, style='CohortType')

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

    @classmethod
    def getHistogramPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.distplot(data[x_feature])

        return gr.figure

    @classmethod
    def getRegPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.regplot(data=data, x=x_feature,
                         y=y_feature)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=gr.get_lines()[0].get_xdata(),
            y=gr.get_lines()[0].get_ydata())
        reg_info = "f(x)={:.3f}x + {:.3f}".format(
            slope, intercept)

        gr.set_title(reg_info)

        return gr.figure

    @classmethod
    def getRegColorPlot(cls, data, x_feature, y_feature, color_by):
        gr = sns.lmplot(data=data, x=x_feature,
                        y=y_feature, hue=color_by, legend_out=True)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=gr.axes.flat[0].get_lines()[0].get_xdata(),
            y=gr.axes.flat[0].get_lines()[0].get_ydata())
        reg_info0 = "f(x)={:.3f}x + {:.3f}".format(
            slope, intercept)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=gr.axes.flat[0].get_lines()[1].get_xdata(),
            y=gr.axes.flat[0].get_lines()[1].get_ydata())
        reg_info1 = "g(x)={:.3f}x + {:.3f}".format(
            slope, intercept)

        reg_info = "{}  |  {}".format(reg_info0, reg_info1)

        gr.fig.suptitle(reg_info)

        return gr

    @classmethod
    def getRegDetailedPlot(cls, data, x_feature, y_feature, color_by):
        def get_stats(x, y):
            """Prints more statistics"""

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=x, y=y)
            reg_info = "f(x)={:.2f}x + {:.2f} \nr^2={:.2f} p={:.2f}".format(
                slope, intercept, r_value, p_value)

            # TODO return value is incompatible with jointplot stat_func
            return reg_info

        def r_squared(x, y):
            return stats.pearsonr(x, y)[0] ** 2

        gr = sns.jointplot(data=data, x=x_feature,
                           y=y_feature, kind="reg", stat_func=r_squared)

        return gr
