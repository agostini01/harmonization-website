import pandas as pd
import numpy as np
from datasets.models import RawUNM
from api.dilutionproc import predict_dilution
from api.analysis import add_confound
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels


def get_dataframe():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""


    # First is necessary to pivot the raw UNM dataset so it matches
    # the requested features.

    # This queries the RawUNM dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawUNM.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    df['creatininemgdl'] = df['creatininemgdl'].astype(float)
    
    df = df[~df['creatininemgdl'].isna()]    

    covars = ['Outcome_weeks', 'age', 'ethnicity', 
                'race', 'education', 'BMI', 'income', 'smoking', 'parity',
                'preg_complications', 'folic_acid_supp', 'fish', 'babySex',
                'birthWt', 'headCirc',
                'birthLen','WeightCentile',
                'LGA','SGA','ga_collection','birth_year']

    df['ga_collection'] = df['gestAge_collection']

    # RAW SAMPLE
    # id PIN_Patient Member_c TimePeriod Analyte    Result  Creat_Corr_Result
    #  1      A0000M        1          1     BCD  1.877245           -99999.0
    #  2      A0001M        1          1     BCD  1.458583           -99999.0
    #  3      A0002M        1          1     BCD  1.694041           -99999.0
    #  4      A0002M        1          1     BCD  1.401296           -99999.0
    #  5      A0003M        1          1     BCD  0.763068           -99999.0

    # Pivoting the table and reseting index
    # TODO - Do we want to plot Result or Creat_Corr_Result
    numerical_values = 'Result'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome'] + covars
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod', 'Outcome'] + covars


    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns)

    df = df.reset_index(level=indexes_to_columns)

    # TODO - Should we drop NaN here?

    df['CohortType'] = 'UNM'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')

    return df

def get_dataframe_BLOD():
    """Returns a pandas DataFrame"""

    # First is necessary to pivot the raw NEU dataset so it matches
    # the requested features.

    # This queries the RawNEU dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawUNM.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )
    
    # Pivoting the table and reseting index
    numerical_values = 'imputed'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome']
    categorical_to_columns = ['Analyte']
   

    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns)
                        
    df = df.reset_index()
   
    # TODO - Should we drop NaN here?

    # After pivot
    # Analyte     TimePeriod Member_c       BCD  ...      UTMO       UTU       UUR
    # PIN_Patient                                ...
    # A0000M               1        1  1.877245  ...  0.315638  1.095520  0.424221
    # A0000M               3        1  1.917757  ...  0.837639  4.549155  0.067877
    # A0001M               1        1  1.458583  ...  0.514317  1.262910  1.554346
    # A0001M               3        1  1.365789  ...  0.143302  1.692582  0.020716
    # A0002M               1        1  1.547669  ...  0.387643  0.988567  1.081877

    df['CohortType'] = 'UNM'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')
    
    return df

def get_dataframe_nofish():
    """Returns a pandas DataFrame with fish removed for cohort"""

    df = get_dataframe()
    df['fish'] = df['fish'].astype(float)
    neu_logic = (df['fish'] == 0)
    df_nofish = df[neu_logic]

    return df_nofish

def get_dataframe_orig():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""


    # First is necessary to pivot the raw UNM dataset so it matches
    # the requested features.

    # This queries the RawUNM dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawUNM.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    df['creatininemgdl'] = df['creatininemgdl'].astype(float)
    
    df = df[~df['creatininemgdl'].isna()]    

    covars = ['Outcome_weeks', 'age', 'ethnicity', 
       'race', 'education', 'BMI', 'income', 'smoking', 'parity',
       'preg_complications', 'folic_acid_supp', 'fish', 'babySex',
       'birthWt', 'headCirc',
       'birthLen','WeightCentile',
       'LGA','SGA','ga_collection','birth_year']

    df['ga_collection'] = df['gestAge_collection']

    # RAW SAMPLE
    # id PIN_Patient Member_c TimePeriod Analyte    Result  Creat_Corr_Result
    #  1      A0000M        1          1     BCD  1.877245           -99999.0
    #  2      A0001M        1          1     BCD  1.458583           -99999.0
    #  3      A0002M        1          1     BCD  1.694041           -99999.0
    #  4      A0002M        1          1     BCD  1.401296           -99999.0
    #  5      A0003M        1          1     BCD  0.763068           -99999.0

    # Pivoting the table and reseting index
    # TODO - Do we want to plot Result or Creat_Corr_Result
    numerical_values = 'Result'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c'] 
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod'] 


    df = pd.pivot_table(df, values=numerical_values,
                            index=columns_to_indexes,
                            columns=categorical_to_columns)

    df = df.reset_index()

    # TODO - Should we drop NaN here?

    # After pivot
    # Analyte     TimePeriod Member_c       BCD  ...      UTMO       UTU       UUR
    # PIN_Patient                                ...
    # A0000M               1        1  1.877245  ...  0.315638  1.095520  0.424221
    # A0000M               3        1  1.917757  ...  0.837639  4.549155  0.067877
    # A0001M               1        1  1.458583  ...  0.514317  1.262910  1.554346
    # A0001M               3        1  1.365789  ...  0.143302  1.692582  0.020716
    # A0002M               1        1  1.547669  ...  0.387643  0.988567  1.081877

    df['CohortType'] = 'UNM'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')

    return df

def get_dataframe_covars():
    # First is necessary to pivot the raw UNM dataset so it matches
    # the requested features.

    # This queries the RawUNM dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawUNM.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    df['creatininemgdl'] = df['creatininemgdl'].astype(float)
    
    df = df[~df['creatininemgdl'].isna()]    

    

    df['ga_collection'] = df['gestAge_collection']

    # RAW SAMPLE
    # id PIN_Patient Member_c TimePeriod Analyte    Result  Creat_Corr_Result
    #  1      A0000M        1          1     BCD  1.877245           -99999.0
    #  2      A0001M        1          1     BCD  1.458583           -99999.0
    #  3      A0002M        1          1     BCD  1.694041           -99999.0
    #  4      A0002M        1          1     BCD  1.401296           -99999.0
    #  5      A0003M        1          1     BCD  0.763068           -99999.0

    df['CohortType'] = 'UNM'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')

    df['Outcome'] = df['Outcome'].astype(float)
    df['LGA'] = df['LGA'].astype(float)
    df['SGA'] = df['SGA'].astype(float)
    

    covars = ['CohortType','TimePeriod','Outcome_weeks', 'age', 'ethnicity', 
       'race', 'education', 'BMI', 'income', 'smoking', 'parity', 'creatininemgdl',
       'preg_complications', 'folic_acid_supp', 'fish', 'babySex',
       'birthWt', 'headCirc', 'Outcome',
       'birthLen','WeightCentile',
       'LGA','SGA','ga_collection','birth_year']

    adjust_types = ['age', 'ethnicity', 
       'race', 'education', 'BMI', 'income', 'smoking', 'parity', 'creatininemgdl',
       'preg_complications', 'folic_acid_supp', 'fish', 'babySex']
    
    for var in adjust_types:
        try:
            df[var] = df[var].astype(float)
        except:
            pass


    ##drop duplicates required because it was initially in long foramt per analyte/visit/participant

    return df[['PIN_Patient'] + covars].drop_duplicates()


def get_dataframe_imputed():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""


    # First is necessary to pivot the raw UNM dataset so it matches
    # the requested features.

    # This queries the RawUNM dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawUNM.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )


    covars = ['Outcome_weeks', 'age', 'ethnicity',
       'race', 'education', 'BMI', 'income', 'smoking', 'parity',
       'preg_complications', 'folic_acid_supp', 'fish', 'babySex',
       'birthWt', 'headCirc',
       'birthLen','WeightCentile',
       'LGA','SGA','ga_collection','birth_year']

    df['ga_collection'] = df['gestAge_collection']

    # RAW SAMPLE
    # id PIN_Patient Member_c TimePeriod Analyte    Result  Creat_Corr_Result
    #  1      A0000M        1          1     BCD  1.877245           -99999.0
    #  2      A0001M        1          1     BCD  1.458583           -99999.0
    #  3      A0002M        1          1     BCD  1.694041           -99999.0
    #  4      A0002M        1          1     BCD  1.401296           -99999.0
    #  5      A0003M        1          1     BCD  0.763068           -99999.0

    # Pivoting the table and reseting index
    # TODO - Do we want to plot Result or Creat_Corr_Result
    numerical_values = 'imputed'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome'] 
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod', 'Outcome'] + covars


    df = pd.pivot_table(df, values=numerical_values,
                            index=columns_to_indexes,
                            columns=categorical_to_columns)

    df = df.reset_index()

    # TODO - Should we drop NaN here?

    # After pivot
    # Analyte     TimePeriod Member_c       BCD  ...      UTMO       UTU       UUR
    # PIN_Patient                                ...
    # A0000M               1        1  1.877245  ...  0.315638  1.095520  0.424221
    # A0000M               3        1  1.917757  ...  0.837639  4.549155  0.067877
    # A0001M               1        1  1.458583  ...  0.514317  1.262910  1.554346
    # A0001M               3        1  1.365789  ...  0.143302  1.692582  0.020716
    # A0002M               1        1  1.547669  ...  0.387643  0.988567  1.081877

    df['CohortType'] = 'UNM'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')

    return df
