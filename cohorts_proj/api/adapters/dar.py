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

    df.columns = ['id', 'unq_id', 'assay',
                  'participant_type', 'time_period', 'batch',
                  'sample_gestage_days', 'preterm', 'urine_specific_gravity',
                  'iAs', 'iAs_IDL', 'iAs_BDL',
                  'AsB', 'AsB_IDL', 'AsB_BDL',
                  'DMA', 'DMA_IDL', 'DMA_BDL',
                  'MMA', 'MMA_IDL', 'MMA_BDL',
                  'Ba', 'Ba_IDL', 'Ba_BDL',
                  'Cs', 'Cs_IDL', 'Cs_BDL',
                  'Sr', 'Sr_IDL', 'Sr_BDL',
                  'Ag', 'Ag_IDL', 'Ag_BDL',
                  'Al', 'Al_IDL', 'Al_BDL',
                  'As', 'As_IDL', 'As_BDL',
                  'Be', 'Be_IDL', 'Be_BDL',
                  'Cd', 'Cd_IDL', 'Cd_BDL',
                  'Co', 'Co_IDL', 'Co_BDL',
                  'Cr', 'Cr_IDL', 'Cr_BDL',
                  'Cu', 'Cu_IDL', 'Cu_BDL',
                  'Fe', 'Fe_IDL', 'Fe_BDL',
                  'Hg', 'Hg_IDL', 'Hg_BDL',
                  'Mn', 'Mn_IDL', 'Mn_BDL',
                  'Mo', 'Mo_IDL', 'Mo_BDL',
                  'Ni', 'Ni_IDL', 'Ni_BDL',
                  'Pb', 'Pb_IDL', 'Pb_BDL',
                  'Sb', 'Sb_IDL', 'Sb_BDL',
                  'Se', 'Se_IDL', 'Se_BDL',
                  'Sn', 'Sn_IDL', 'Sn_BDL',
                  'Tl', 'Tl_IDL', 'Tl_BDL',
                  'U', 'U_IDL', 'U_BDL',
                  'W', 'W_IDL', 'W_BDL',
                  'Zn', 'Zn_IDL', 'Zn_BDL',
                  'V', 'V_IDL', 'V_BDL'],

    return df
