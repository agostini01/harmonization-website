from rest_framework import generics, views
from django.http import HttpResponse

from .serializers import DatasetUploadSerializer
from .models import DatasetUploadModel

from .serializers import GraphRequestSerializer

from datasets.models import RawFlower, RawUNM, RawNEU, RawDAR

from api import adapters
from api import graphs
from api import analysis
import os

import numpy as np

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

    df = df.replace(np.nan, None, regex=True)

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
            Outcome_weeks = entry.gestAge,
            age = entry.age,
            ethnicity = entry.ethnicity,
            race = entry.race,
            education = entry.education,
            BMI = entry.BMI,
            income = entry.income,
            smoking = entry.smoking,
            parity = entry.parity,
            preg_complications	= entry.preg_complications,
            folic_acid_supp	= entry.folic_acid_supp,
            fish = entry.fish,
            babySex	= entry.babySex,
            birthWt = entry.birthWt,
            birthLen = entry.birthLen,
            WeightZScore = entry.WeightZScore,
            WeightCentile = entry.WeightCentile,
            birth_year = entry.birth_year,
            birth_month = entry.birth_month,
            LGA	= entry.LGA,
            SGA= entry.SGA,
            gestAge_collection = entry.gestAge_collection,
            headCirc = entry.headCir
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
    df = df.replace(np.nan, None, regex=True)
    # Delete database
    RawNEU.objects.all().delete()

    for entry in df.itertuples():
        entry = RawNEU.objects.create(
            PIN_Patient=entry.studyid,
            Member_c=entry.Member_c,
            TimePeriod=entry.TimePeriod,
            LOD=entry.LOD,
            Analyte=entry.Analyte,
            Result=entry.Result,
            Outcome=entry.PretermBirth,
            Outcome_weeks = entry.gestationAge,
            age = entry.isage,
            ethnicity = entry.hisporg,
            race = entry.race,
            ed = entry.ed,
            BMI = entry.BMI,
            fvinc =entry.fvinc,
            smoking =entry.smoking,
            pregnum = entry.pregnum,
            preg_complications	=entry.preg_complications,
            folic_acid_supp	=entry.folic,
            fish =entry.fish,
            babySex	=entry.babySex,
            birthWt =entry.weightkg,
            birthLen =entry.ppnblength,
            headCirc = entry.ppheadcircumference,
            fvdate = entry.fvdate,
            svdate = entry.svdate,
            tvdate = entry.tvdate,
            SPECIFICGRAVITY_V1 = entry.SPECIFICGRAVITY_V1,
            SPECIFICGRAVITY_V2 = entry.SPECIFICGRAVITY_V2,
            SPECIFICGRAVITY_V3 = entry.SPECIFICGRAVITY_V3,
            WeightZScore = entry.weightzscore,
            WeightCentile = entry.weightcentile,
            LGA	= entry.lga,
            SGA= entry.sga,
            PPDATEDEL = entry.PPDATEDEL,
            ga_collection = entry.ga_collection,
            fish_pu_v1 = entry.fish_pu_v1,
            fish_pu_v2 = entry.fish_pu_v2,
            fish_pu_v3 = entry.fish_pu_v3

        )

def saveDARToDB(csv_file):
    df = pd.read_csv(csv_file,
                     skip_blank_lines=True,
                     header=0)

    df = df.replace(np.nan, None, regex=True)

    # Delete database
    RawDAR.objects.all().delete()

    for entry in df.itertuples():
        entry = RawDAR.objects.create(
            unq_id=entry.unq_id,
            sample_gestage_days= entry.gestAge,
            preterm = entry.pretermBirth,
            age = entry.age,
            ethnicity = entry.ethnicity,
            race =  entry.race,
            education = entry.education,
            BMI = entry.BMI,
            smoking =  entry.smoking,
            parity =   entry.parity,
            babySex = entry.babySex,
            headCirc = entry.headCirc,
            birthLen = entry.birthLen,
            birthWt =  entry.birthWt,
            ponderal =  entry.ponderal,
            preg_complications = entry.preg_complications,
            birth_year = entry.birth_year,
            birth_month = entry.birth_month,
            PNFFQTUNA = entry.PNFFQTUNA,
            PNFFQFR_FISH_KIDS = entry.PNFFQFR_FISH_KIDS,
            PNFFQSHRIMP_CKD = entry.PNFFQSHRIMP_CKD,
            PNFFQDK_FISH = entry.PNFFQDK_FISH,
            PNFFQOTH_FISH = entry.PNFFQOTH_FISH,
            mfsp_6 = entry.mfsp_6,
            fish = entry.fish,
            TOTALFISH_SERV = entry.TOTALFISH_SERV,
            folic_acid = entry.folic_acid,
            income5y = entry.income5y,
            time_period = entry.time_period,
            participant_type=entry.participant_type,
            urine_batchno_bulk =entry.urine_batchno_bulk, 
            urine_batchno_spec=entry.urine_batchno_spec,
            collect_age_days=entry.urine_m24G_collect_age_days,
            collect_age_src=entry.urine_m24G_collect_age_src,
            collection_season=entry.urine_m24G_collection_season,
            urine_specific_gravity=entry.urine_m24G_Specific_Gravity,
            pH=entry.urine_m24G_pH,
            TotAs_noAsB=entry.urine_m24G_TotAs_noAsB,
            PropMMAtoiAs=entry.urine_m24G_PropMMAtoiAs,
            PropDMAtoMMA=entry.urine_m24G_PropDMAtoMMA,
            # squid=entry.squid,
            WeightZScore = entry.WeightZScore,
            WeightCentile = entry.WeightCentile,
            LGA	= entry.LGA,
            SGA= entry.SGA,
            Ag=entry.urine_m24G_Ag,
            Ag_BDL=entry.urine_m24G_bdl_Ag,
            Ag_IDL=entry.urine_m24G_dl_Ag,
            Ag_N=entry.urine_m24G_Ag_n,
            Al=entry.urine_m24G_Al,
            Al_BDL=entry.urine_m24G_bdl_Al,
            Al_IDL=entry.urine_m24G_dl_Al,
            Al_N=entry.urine_m24G_Al_n,
            As=entry.urine_m24G_As,
            As_BDL=entry.urine_m24G_bdl_As,
            As_IDL=entry.urine_m24G_dl_As,
            As_N=entry.urine_m24G_As_n,
            AsB=entry.urine_m24G_AsB,
            AsB_BDL=entry.urine_m24G_bdl_AsB,
            AsB_IDL=entry.urine_m24G_dl_AsB,
            AsB_N=entry.urine_m24G_AsB_n,
            iAs=entry.urine_m24G_iAs,
            iAs_BDL=entry.urine_m24G_bdl_iAs,
            iAs_IDL=entry.urine_m24G_dl_iAs,
            iAs_N=entry.urine_m24G_iAs_n,
            AsIII=entry.urine_m24G_AsIII,
            AsIII_BDL=entry.urine_m24G_bdl_AsIII,
            AsIII_IDL=entry.urine_m24G_dl_AsIII,
            AsIII_N=entry.urine_m24G_AsIII_n,
            AsV=entry.urine_m24G_AsV,
            AsV_BDL=entry.urine_m24G_bdl_AsV,
            AsV_IDL=entry.urine_m24G_dl_AsV,
            AsV_N=entry.urine_m24G_AsV_n,
            Ba=entry.urine_m24G_Ba,
            Ba_BDL=entry.urine_m24G_bdl_Ba,
            Ba_IDL=entry.urine_m24G_dl_Ba,
            Ba_N=entry.urine_m24G_Ba_n,
            Be=entry.urine_m24G_Be,
            Be_BDL=entry.urine_m24G_bdl_Be,
            Be_IDL=entry.urine_m24G_dl_Be,
            Be_N=entry.urine_m24G_Be_n,
            Ca=entry.urine_m24G_Ca,
            Ca_BDL=entry.urine_m24G_bdl_Ca,
            Ca_IDL=entry.urine_m24G_dl_Ca,
            Ca_N=entry.urine_m24G_Ca_n,
            Cd=entry.urine_m24G_Cd,
            Cd_BDL=entry.urine_m24G_bdl_Cd,
            Cd_IDL=entry.urine_m24G_dl_Cd,
            Cd_N=entry.urine_m24G_Cd_n,
            Co=entry.urine_m24G_Co,
            Co_BDL=entry.urine_m24G_bdl_Co,
            Co_IDL=entry.urine_m24G_dl_Co,
            Co_N=entry.urine_m24G_Co_n,
            Cr=entry.urine_m24G_Cr,
            Cr_BDL=entry.urine_m24G_bdl_Cr,
            Cr_IDL=entry.urine_m24G_dl_Cr,
            Cr_N=entry.urine_m24G_Cr_n,
            Cs=entry.urine_m24G_Cs,
            Cs_BDL=entry.urine_m24G_bdl_Cs,
            Cs_IDL=entry.urine_m24G_dl_Cs,
            Cs_N=entry.urine_m24G_Cs_n,
            Cu=entry.urine_m24G_Cu,
            Cu_BDL=entry.urine_m24G_bdl_Cu,
            Cu_IDL=entry.urine_m24G_dl_Cu,
            Cu_N=entry.urine_m24G_Cu_n,
            DMA=entry.urine_m24G_DMA,
            DMA_BDL=entry.urine_m24G_bdl_DMA,
            DMA_IDL=entry.urine_m24G_dl_DMA,
            DMA_N=entry.urine_m24G_DMA_n,
            Fe=entry.urine_m24G_Fe,
            Fe_BDL=entry.urine_m24G_bdl_Fe,
            Fe_IDL=entry.urine_m24G_dl_Fe,
            Fe_N=entry.urine_m24G_Fe_n,
            Hg=entry.urine_m24G_Hg,
            Hg_BDL=entry.urine_m24G_bdl_Hg,
            Hg_IDL=entry.urine_m24G_dl_Hg,
            Hg_N=entry.urine_m24G_Hg_n,
            K=entry.urine_m24G_K,
            K_BDL=entry.urine_m24G_bdl_K,
            K_IDL=entry.urine_m24G_dl_K,
            K_N=entry.urine_m24G_K_n,
            Mg=entry.urine_m24G_Mg,
            Mg_BDL=entry.urine_m24G_bdl_Mg,
            Mg_IDL=entry.urine_m24G_dl_Mg,
            Mg_N=entry.urine_m24G_Mg_n,
            MMA=entry.urine_m24G_MMA,
            MMA_BDL=entry.urine_m24G_bdl_MMA,
            MMA_IDL=entry.urine_m24G_dl_MMA,
            MMA_N=entry.urine_m24G_MMA_n,
            Mn=entry.urine_m24G_Mn,
            Mn_BDL=entry.urine_m24G_bdl_Mn,
            Mn_IDL=entry.urine_m24G_dl_Mn,
            Mn_N=entry.urine_m24G_Mn_n,
            Mo=entry.urine_m24G_Mo,
            Mo_BDL=entry.urine_m24G_bdl_Mo,
            Mo_IDL=entry.urine_m24G_dl_Mo,
            Mo_N=entry.urine_m24G_Mo_n,
            Ni=entry.urine_m24G_Ni,
            Ni_BDL=entry.urine_m24G_bdl_Ni,
            Ni_IDL=entry.urine_m24G_dl_Ni,
            Ni_N=entry.urine_m24G_Ni_n,
            P=entry.urine_m24G_P,
            P_BDL=entry.urine_m24G_bdl_P,
            P_IDL=entry.urine_m24G_dl_P,
            P_N=entry.urine_m24G_P_n,
            Pb=entry.urine_m24G_Pb,
            Pb_BDL=entry.urine_m24G_bdl_Pb,
            Pb_IDL=entry.urine_m24G_dl_Pb,
            Pb_N=entry.urine_m24G_Pb_n,
            Sb=entry.urine_m24G_Sb,
            Sb_BDL=entry.urine_m24G_bdl_Sb,
            Sb_IDL=entry.urine_m24G_dl_Sb,
            Sb_N=entry.urine_m24G_Sb_n,
            Se=entry.urine_m24G_Se,
            Se_BDL=entry.urine_m24G_bdl_Se,
            Se_IDL=entry.urine_m24G_dl_Se,
            Se_N=entry.urine_m24G_Se_n,
            Si=entry.urine_m24G_Si,
            Si_BDL=entry.urine_m24G_bdl_Si,
            Si_IDL=entry.urine_m24G_dl_Si,
            Si_N=entry.urine_m24G_Si_n,
            Sn=entry.urine_m24G_Sn,
            Sn_BDL=entry.urine_m24G_bdl_Sn,
            Sn_IDL=entry.urine_m24G_dl_Sn,
            Sn_N=entry.urine_m24G_Sn_n,
            Sr=entry.urine_m24G_Sr,
            Sr_BDL=entry.urine_m24G_bdl_Sr,
            Sr_IDL=entry.urine_m24G_dl_Sr,
            Sr_N=entry.urine_m24G_Sr_n,
            Tl=entry.urine_m24G_Tl,
            Tl_BDL=entry.urine_m24G_bdl_Tl,
            Tl_IDL=entry.urine_m24G_dl_Tl,
            Tl_N=entry.urine_m24G_Tl_n,
            U=entry.urine_m24G_U,
            U_BDL=entry.urine_m24G_bdl_U,
            U_IDL=entry.urine_m24G_dl_U,
            U_N=entry.urine_m24G_U_n,
            V=entry.urine_m24G_V,
            V_BDL=entry.urine_m24G_bdl_V,
            V_IDL=entry.urine_m24G_dl_V,
            V_N=entry.urine_m24G_V_n,
            W=entry.urine_m24G_W,
            W_BDL=entry.urine_m24G_bdl_W,
            W_IDL=entry.urine_m24G_dl_W,
            W_N=entry.urine_m24G_W_n,
            Zn=entry.urine_m24G_Zn,
            Zn_BDL=entry.urine_m24G_bdl_Zn,
            Zn_IDL=entry.urine_m24G_dl_Zn,
            Zn_N=entry.urine_m24G_Zn_n
        )




class DatasetUploadView(generics.CreateAPIView):
    """Handles only POST methods."""
    serializer_class = DatasetUploadSerializer
    queryset = DatasetUploadModel.objects.all()

    def post(self, request, *args, **kwargs):
        """Saves CSV to DatasetModel database and populate raw databases."""
        
        if request.data['dataset_type'] != 'csv_only':

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

        req = self.getPlot(request)
        
        return req

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
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        if dataset_type == 'neudar_dataset':
            df1 = adapters.neu.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        if dataset_type == 'darunm_dataset':
            df1 = adapters.unm.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        # This is the harmonized dataset
        if dataset_type == 'har_dataset':
            # TODO Handle early exit when selected columns are not present
            # selected_columns = [x_feature, y_feature, color_by, 'CohortType']
            df1 = adapters.unm.get_dataframe()  # [selected_columns]
            df2 = adapters.neu.get_dataframe()  # [selected_columns]
            df3 = adapters.dar.get_dataframe()  # [selected_columns]
            #df = pd.concat([df1, df2, df3])
            df = analysis.merge3CohortFrames(df1, df2, df3)
            
        # Apply Filters
        if time_period != 9:
            df = df[df['TimePeriod'] == time_period]
        else:
            pass

        # Build response figure
        response = HttpResponse(content_type="image/jpg")
        plt.clf()

        graph_options = ['scatter_plot','individual_scatter_plot','corr_plot', 'clustermap','analysis', 'covars_facet_continous', \
            'arsenic_facet_continous','covars_facet_categorical', 'custom_facet_LM_plot','pair_plot','cat_plot', 'violin_cat_plot', \
                'histogram_plot','kde_plot','linear_reg_plot','linear_reg_with_color_plot']
                
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
                    
            if (t == 'corr_plot'):
                gr = cls.getCorrelationHeatmap(
                    df)
            
            if (t == 'clustermap'):
                gr = cls.getClusterMap(
                    df, color_by)

            if (t == 'analysis'):
                print(os.listdir())
                analysis.runcustomanalysis()
                gr = cls.getIndividualScatterPlot(
                    df, x_feature, y_feature, color_by)

            if (t == 'covars_facet_continous'):

                gr = cls.getCustomFacetContinuousPlot1(
                df, x_feature, y_feature, color_by, 0)
            
            if (t == 'arsenic_facet_continous'):

                gr = cls.getCustomFacetContinuousPlot1(
                df, x_feature, y_feature, color_by, 1)

            if (t == 'covars_facet_categorical'):
                
                gr = cls.getCustomFacetCategoricalPlot1(
                df, x_feature, y_feature, color_by)
            
            if (t == 'custom_facet_LM_plot'):
                gr = cls.getCustomFacetLMPlot1(
                df, x_feature, y_feature, color_by)

            if (t == 'pair_plot'):
                gr = cls.getPairPlot(df, x_feature, y_feature, color_by)

            if (t == 'cat_plot'):
                gr = cls.getCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'violin_cat_plot'):
                gr = cls.getViolinCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'histogram_plot'):
                gr = cls.getHistogramPlot(df, x_feature, y_feature, color_by)
            
            if (t == 'kde_plot'):
                gr = cls.getKdePlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_plot'):
                gr = cls.getRegPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_with_color_plot'):
                gr = cls.getRegColorPlot(df, x_feature, y_feature, color_by)

           # if (t == 'linear_reg_detailed_plot'):
            #    gr = cls.getRegDetailedPlot(df, x_feature, y_feature, color_by)

            #if (t == 'linear_mixed_ml_summary'):
            #    #gr = cls.getMLPlot(df, x_feature, y_feature, color_by)
            #    gr = ''

            #if (t == 'logistic_regression'):
            #    gr = cls.getlogistcRegPlot(df, x_feature, y_feature, color_by)
            
            if (t not in graph_options):
                
                gr = graphs.noGraphMessage()
                gr.savefig(response, format="jpg", dpi=fig_dpi)
                
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
    def getCustomFacetContinuousPlot1(cls, data, x_feature, y_feature, time_period, type,
                                 info=True):
        if info:
            return graphs.getCustomFacetContinuousPlot1(
                data, x_feature, y_feature, time_period, type)
        else:
            return graphs.getCustomFacetContinuousPlot1(
                data, x_feature, y_feature, time_period, type)

    @classmethod
    def getCustomFacetCategoricalPlot1(cls, data, x_feature, y_feature, time_period,
                                 info=True):
        if info:
            return graphs.getCustomFacetCategoricalPlot1(
                data, x_feature, y_feature, time_period)
        else:
            return graphs.getCustomFacetCategoricalPlot1(
                data, x_feature, y_feature, time_period)
    
    @classmethod
    def getCustomFacetLMPlot1(cls, data, x_feature, y_feature, time_period,
                                 info=True):
        if info:
            return graphs.getCustomFacetLMPlot1(
                data, x_feature, y_feature, time_period)
        else:
            return graphs.getCustomFacetLMPlot1(
                data, x_feature, y_feature, time_period)

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
            return graphs.getViolinCatPlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getViolinCatPlotWithInfo(
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
    def getKdePlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getKdePlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getKdePlot(
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
    
    @classmethod
    def getCorrelationHeatmap(cls, data):
        
        return graphs.getCorrelationHeatmap(data)

    @classmethod
    def getClusterMap(cls, data, color_by):
        
        return graphs.getClusterMap(data, color_by)
    @classmethod
    def getMLPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getMLPlot(data, x_feature, y_feature, color_by)

    @classmethod
    def getbinomialMLPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getbinomialMLPlot(data, x_feature, y_feature, color_by)

    @classmethod
    def getlogistcRegPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getlogistcRegPlot(data, x_feature, y_feature, color_by)

class InfoRequestView(views.APIView):
    """Handles only POST methods."""
    serializer_class = GraphRequestSerializer
    # queryset = ()

    """
    Concrete view for listing a queryset or creating a model instance.
    """
 
    def get(self, request, *args, **kwargs):

        req = self.getPlot(request)
        
        return req

    @classmethod
    def getPlot(cls, request):
        """Called during get request to generate plots."""
        print('Request data')
        print(request.data)

        plot_type = request.data['plot_type']
        x_feature = request.data['x_feature']
        y_feature = request.data['y_feature']
        color_by = request.data['color_by']
        time_period = int(request.data['time_period'])
        fig_dpi = int(request.data['fig_dpi'])
        dataset_type = request.data['dataset_type']
        covar_choices = request.data['covar_choices']
        adjust_dilution = request.data['adjust_dilution']

        print('### covar choices ###')
        print(covar_choices)
        
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
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        if dataset_type == 'neudar_dataset':
            df1 = adapters.neu.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        if dataset_type == 'darunm_dataset':
            df1 = adapters.unm.get_dataframe()
            df2 = adapters.dar.get_dataframe()
            #df = pd.concat([df1, df2])
            df = analysis.merge2CohortFrames(df1, df2)

        # This is the harmonized dataset
        if dataset_type == 'har_dataset':
            # TODO Handle early exit when selected columns are not present
            # selected_columns = [x_feature, y_feature, color_by, 'CohortType']
            df1 = adapters.unm.get_dataframe()  # [selected_columns]
            df2 = adapters.neu.get_dataframe()  # [selected_columns]
            df3 = adapters.dar.get_dataframe()  # [selected_columns]
            #df = pd.concat([df1, df2, df3])
            df = analysis.merge3CohortFrames(df1, df2, df3)
            

        print('((((((')   
        print(time_period)
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
                    
            if (t == 'corr_plot'):
                gr = cls.getCorrelationHeatmap(
                    df)
            
            if (t == 'clustermap'):
                gr = cls.getClusterMap(
                    df, color_by)

            if (t == 'analysis'):
                print(os.listdir())
                analysis.runcustomanalysis()
                gr = cls.getIndividualScatterPlot(
                    df, x_feature, y_feature, color_by)

            if (t == 'covars_facet_continous'):

                #gr = cls.getCustomFacetContinuousPlot1(
                #df, x_feature, y_feature, color_by, 0)

                gr = str(df[[x_feature,y_feature]].describe().to_html())
            
            if (t == 'arsenic_facet_continous'):

                gr = cls.getCustomFacetContinuousPlot1(
                df, x_feature, y_feature, color_by, 1)

            if (t == 'covars_facet_categorical'):
                
                gr = cls.getCustomFacetCategoricalPlot1(
                df, x_feature, y_feature, color_by)
            
            if (t == 'custom_facet_LM_plot'):
                gr = cls.getCustomFacetLMPlot1(
                df, x_feature, y_feature, color_by)

            if (t == 'pair_plot'):
                gr = cls.getPairPlot(df, x_feature, y_feature, color_by)

            if (t == 'cat_plot'):
                gr = cls.getCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'violin_cat_plot'):
                gr = cls.getViolinCatPlot(df, x_feature, y_feature, color_by)

            if (t == 'histogram_plot'):
                gr = cls.getHistogramPlot(df, x_feature, y_feature, color_by)
            
            if (t == 'kde_plot'):
                gr = cls.getKdePlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_plot'):
                gr = cls.getRegPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_with_color_plot'):
                gr = cls.getRegColorPlot(df, x_feature, y_feature, color_by)

            if (t == 'linear_reg_detailed_plot'):
                gr = analysis.crude_reg(df, x_feature, y_feature, covar_choices, adjust_dilution, 'html')

            if (t == 'linear_mixed_ml_summary'):
                gr = analysis.crude_mixedML2(df, x_feature, y_feature, covar_choices)
                
            if (t == 'binomial_mixed_ml_summary'):
                gr = analysis.crude_binomial_mixedML(df, x_feature, y_feature, include_covars)
                gr = gr.as_html()

            if (t == 'logistic_regression'):
                gr = analysis.crude_logreg(df, x_feature, y_feature, covar_choices, adjust_dilution, 'html')
            
            if (t == 'categorical_summary'):
                gr = analysis.categoricalCounts(df).to_html()
            
            if (t == 'continous_summary'):
                gr = analysis.cohortdescriptive_all(df).to_html()
            
            if (t == 'bayesian_mixed_ml'):
                gr = analysis.crude_mixedMLbayse(df, x_feature, y_feature, include_covars, False).to_html()

            if (t == 'binomial_bayesian_mixed_ml'):
                gr = analysis.crude_mixedMLbayse(df, x_feature, y_feature, include_covars, True).to_html()

            if (t == 'custom_analysis'):
                analysis.runcustomanalysis()
                
                gr = graphs.noDataMessage()


        
        response = HttpResponse(gr)

    
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
    def getCustomFacetContinuousPlot1(cls, data, x_feature, y_feature, time_period, type,
                                 info=True):
        if info:
            return graphs.getCustomFacetContinuousPlot1(
                data, x_feature, y_feature, time_period, type)
        else:
            return graphs.getCustomFacetContinuousPlot1(
                data, x_feature, y_feature, time_period, type)

    @classmethod
    def getCustomFacetCategoricalPlot1(cls, data, x_feature, y_feature, time_period,
                                 info=True):
        if info:
            return graphs.getCustomFacetCategoricalPlot1(
                data, x_feature, y_feature, time_period)
        else:
            return graphs.getCustomFacetCategoricalPlot1(
                data, x_feature, y_feature, time_period)
    
    @classmethod
    def getCustomFacetLMPlot1(cls, data, x_feature, y_feature, time_period,
                                 info=True):
        if info:
            return graphs.getCustomFacetLMPlot1(
                data, x_feature, y_feature, time_period)
        else:
            return graphs.getCustomFacetLMPlot1(
                data, x_feature, y_feature, time_period)

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
            return graphs.getViolinCatPlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getViolinCatPlotWithInfo(
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
    def getKdePlot(cls, data, x_feature, y_feature, color_by, info=True):
        if info:
            return graphs.getKdePlotWithInfo(
                data, x_feature, y_feature, color_by)
        else:
            return graphs.getKdePlot(
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
    
    @classmethod
    def getCorrelationHeatmap(cls, data):
        
        return graphs.getCorrelationHeatmap(data)

    @classmethod
    def getClusterMap(cls, data, color_by):
        
        return graphs.getClusterMap(data, color_by)
    @classmethod
    def getMLPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getMLPlot(data, x_feature, y_feature, color_by)

    @classmethod
    def getbinomialMLPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getbinomialMLPlot(data, x_feature, y_feature, color_by)

    @classmethod
    def getlogistcRegPlot(cls, data, x_feature, y_feature, color_by):
        return graphs.getlogistcRegPlot(data, x_feature, y_feature, color_by)


