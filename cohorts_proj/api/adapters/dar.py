import pandas as pd
import numpy as np
from datasets.models import RawDAR


def get_dataframe():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""

    # First is necessary to pivot the raw DAR dataset so it matches
    # the requested features.

    # This queries the RawDAR dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawDAR.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    df.columns = ['id', 'PIN_Patient', 'assay',
                  'Member_c', 'TimePeriod', 
                  'sample_gestage_days', 'Outcome', 
                  'age','ethnicity','race','education','BMI',
                  'smoking','parity', 'preg_complications',
                  'folic_acid_supp','babySex',
                  'birthWt','birthLen','headCirc','ponderal','PNFFQTUNA',
                  'PNFFQFR_FISH_KIDS','PNFFQSHRIMP_CKD','PNFFQDK_FISH','PNFFQOTH_FISH',
                  'mfsp_6','TOTALFISH_SERV','fish','folic_acid','income5y','urine_batchno_bulk','urine_batchno_spec',
                  'collect_age_days','collect_age_src','collection_season',
                  'pH','UTAS','PropMMAtoiAs','PropDMAtoMMA','urine_specific_gravity',
                  'WeightZScore',
                    'WeightCentile',
                    'LGA',
                    'SGA',
                  'UIAS', 'iAs_IDL', 'iAs_BDL','iAs_N',
                  'UASB', 'AsB_IDL', 'AsB_BDL','AsB__N',
                  'UAS3', 'AsIII_IDL', 'AsIII_BDL','AsIII_N',
                  'UAS5', 'AsV_IDL', 'AsV_BDL','AsV_N',
                  'UDMA', 'DMA_IDL', 'DMA_BDL','DMA_N',
                  'UMMA', 'MMA_IDL', 'MMA_BDL','MMA_N',
                  'UBA', 'Ba_IDL', 'Ba_BDL','Ba_N',
                  'UAG', 'Ag_IDL', 'Ag_BDL','Ag_N',
                  'UAL', 'Al_IDL', 'Al_BDL','Al_N',
                  'UAS', 'As_IDL', 'As_BDL','As_N',
                  'UBE', 'Be_IDL', 'Be_BDL', 'Be_N',
                  'UCA', 'Ca_IDL', 'Ca_BDL','Ca_N',
                  'UCD', 'Cd_IDL', 'Cd_BDL','Cd_N',
                  'UCO', 'Co_IDL', 'Co_BDL','Co_N',
                  'UCR', 'Cr_IDL', 'Cr_BDL','Cr_N',
                  'UCS', 'Cs_IDL', 'Cs_BDL','Cs_N',
                  'UCU', 'Cu_IDL', 'Cu_BDL','Cu_N',
                  'UFE', 'Fe_IDL', 'Fe_BDL','Fe_N',
                  'UHG', 'Hg_IDL', 'Hg_BDL','Hg_N',
                  'UPO', 'K_IDL', 'K_BDL','K_N',
                  'UMG', 'Mg_IDL', 'Mg_BDL','Mg_N',
                  'UMN', 'Mn_IDL', 'Mn_BDL','Mn_N',
                  'UMO', 'Mo_IDL', 'Mo_BDL','Mo_N',
                  'UNI', 'Ni_IDL', 'Ni_BDL','Ni_N',
                  'UPP', 'P_IDL', 'P_BDL','P_N',
                  'UPB', 'Pb_IDL', 'Pb_BDL','Pb_N',
                  'USB', 'Sb_IDL', 'Sb_BDL','Sb_N',
                  'USE', 'Se_IDL', 'Se_BDL', 'Se_N', 
                  'USI', 'Si_IDL', 'Si_BDL', 'Si_N', 
                  'USN', 'Sn_IDL', 'Sn_BDL', 'Sn_N',
                  'USR', 'Sr_IDL', 'Sr_BDL', 'Sr_N',
                  'UTL', 'Tl_IDL', 'Tl_BDL','Tl_N',
                  'UUR', 'U_IDL', 'U_BDL', 'U_N',
                  'UTU', 'W_IDL', 'W_BDL','W_N',
                  'UZN', 'Zn_IDL', 'Zn_BDL','Zn_N',
                  'UVA', 'V_IDL', 'V_BDL','V_N']

    df['Outcome_weeks'] = df['sample_gestage_days'] 

    df['ga_collection'] = df['collect_age_days'] /7 

    # Read numeric columns as numeric
    numeric_columns =     ['UIAS', 
                  'UASB', 
                  'UAS3', 
                  'UAS5', 
                  'UDMA', 
                  'UMMA', 
                  'UBA', 
                  'UAG', 
                  'UAL', 
                  'UAS', 
                  'UBE', 
                  'UCA', 
                  'UCD', 
                  'UCO', 
                  'UCR', 
                  'UCS', 
                  'UCU', 
                  'UFE', 
                  'UHG', 
                  'UPO', 
                  'UMG', 
                  'UMN', 
                  'UMO', 
                  'UNI', 
                  'UPP', 
                  'UPB', 
                  'USB', 
                  'USE', 
                  'USI', 
                  'USN', 
                  'USR', 
                  'UTL', 
                  'UUR', 
                  'UTU', 
                  'UZN', 
                  'UVA']
                       
    for c in numeric_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # # To convert from mg/dL to ug/L we must multiply by 10,000
    # analytes_for_mg_dl_to_ug_L = [
    # ]

    # for a in analytes_for_mg_dl_to_ug_L:
    #     df[a]=df[a]*10000

    df['CohortType'] = 'DAR'

    time_period_mapper = {
        '12G': 0,
        '24G': 1,
        '6WP': 3,
        '6MP': 8,
        '1YP': 8,
        '2YP': 8,
        '3YP': 8,
        '5YP': 8,
    }
    df['TimePeriod'] = df['TimePeriod'].map(time_period_mapper)

    return df