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
                  'Member_c', 'TimePeriod', 'batch',
                  'sample_gestage_days', 'Outcome', 'urine_specific_gravity',
                  'UTAS', 'iAs_IDL', 'iAs_BDL',
                  'UASB', 'AsB_IDL', 'AsB_BDL',
                  'UDMA', 'DMA_IDL', 'DMA_BDL',
                  'UMMA', 'MMA_IDL', 'MMA_BDL',
                  'UBA', 'Ba_IDL', 'Ba_BDL',
                  'UCS', 'Cs_IDL', 'Cs_BDL',
                  'USR', 'Sr_IDL', 'Sr_BDL',
                  'UAG', 'Ag_IDL', 'Ag_BDL',
                  'UAL', 'Al_IDL', 'Al_BDL',
                  'UAS3', 'As_IDL', 'As_BDL',
                  'UBE', 'Be_IDL', 'Be_BDL',
                  'UCD', 'Cd_IDL', 'Cd_BDL',
                  'UCO', 'Co_IDL', 'Co_BDL',
                  'UCR', 'Cr_IDL', 'Cr_BDL',
                  'UCU', 'Cu_IDL', 'Cu_BDL',
                  'UFE', 'Fe_IDL', 'Fe_BDL',
                  'UHG', 'Hg_IDL', 'Hg_BDL',
                  'UMN', 'Mn_IDL', 'Mn_BDL',
                  'UMO', 'Mo_IDL', 'Mo_BDL',
                  'UNI', 'Ni_IDL', 'Ni_BDL',
                  'UPB', 'Pb_IDL', 'Pb_BDL',
                  'USB', 'Sb_IDL', 'Sb_BDL',
                  'USE', 'Se_IDL', 'Se_BDL',  # Missing?
                  'USN', 'Sn_IDL', 'Sn_BDL',
                  'UTL', 'Tl_IDL', 'Tl_BDL',
                  'UUR', 'U_IDL', 'U_BDL',
                  'UTU', 'W_IDL', 'W_BDL',
                  'UZN', 'Zn_IDL', 'Zn_BDL',
                  'UVA', 'V_IDL', 'V_BDL']

    # Read numeric columns as numeric
    numeric_columns = ['UTAS', 'UASB', 'UDMA',
                       'UMMA', 'UBA', 'UCS',
                       'USR', 'UAG', 'UAL',
                       'UAS3', 'UBE', 'UCD',
                       'UCO', 'UCR', 'UCU',
                       'UFE', 'UHG', 'UMN',
                       'UMO', 'UNI', 'UPB',
                       'USB', 'USE', 'USN',
                       'UTL', 'UUR', 'UTU',
                       'UZN', 'UVA' ]

    for c in numeric_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # # To convert from mg/dL to ug/L we must multiply by 10,000
    # analytes_for_mg_dl_to_ug_L = [
    # ]

    # for a in analytes_for_mg_dl_to_ug_L:
    #     df[a]=df[a]*10000

    df['CohortType'] = 'Dartmouth'

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
