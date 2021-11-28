import pandas as pd
import numpy as np
from datasets.models import RawNEU
from api.dilutionproc import predict_dilution
from api.analysis import add_confound
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels


def get_dataframe():
    """Returns a pandas DataFrame with the correct
    format for the generic plotting functions."""

    # First is necessary to pivot the raw NEU dataset so it matches
    # the requested features.

    # This queries the RawNEU dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawNEU.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    ## birth weight and length
  
    df['birthWt'] = df['birthWt'] * 1000
    df['birthLen'] = df['birthLen'] * 2.54

    ## education connversion
    df['ed'] = df['ed'].astype(float)

    conditions = [
    (df['ed'] <= 8 ),
    (df['ed'].isin([9,10,11,12])),
    (df['ed'].isin([13,14])) ,
    (df['ed'] == 15 ),
    (df['ed'] > 15 ) &  (df['ed'] < 20) ,
    ]
    choices = [1,2,3,4,5]

    df['education'] = np.select(conditions, choices, default=-9)
    ## birth year
    df['PPDATEDEL'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce')
    df['birth_year'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce').dt.year
    
    ## new covariates

    ## df.rename(columns = {'pregnum':'parity'}, inplace = True)
    #new covars

    covars = ['Outcome_weeks', 'age', 'ethnicity', 'race', 
    'BMI', 'smoking', 'parity', 'preg_complications',
    'folic_acid_supp', 'fish', 'babySex', 'birthWt', 'birthLen', 'headCirc',
    'WeightCentile','LGA','SGA','ga_collection','education', 'birth_year', 
    'SPECIFICGRAVITY_V2', 'fish_pu_v2']

    #calculate extra variables
    #parity
    df['parity'] = df['pregnum']
    #ga at collection

    # Pivoting the table and reseting index
    numerical_values = 'Result'
    # TODO - Fix this since Covars can have NaN's
    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome'] + covars
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod', 'Outcome'] + covars

    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns,
                        aggfunc=np.average)
                        
    df = df.reset_index(level=indexes_to_columns)
 
    # TODO - Should we drop NaN here?

    # After pivot
    # Analyte     TimePeriod Member_c       BCD  ...      UTMO       UTU       UUR
    # PIN_Patient                                ...
    # A0000M               1        1  1.877245  ...  0.315638  1.095520  0.424221
    # A0000M               3        1  1.917757  ...  0.837639  4.549155  0.067877
    # A0001M               1        1  1.458583  ...  0.514317  1.262910  1.554346
    # A0001M               3        1  1.365789  ...  0.143302  1.692582  0.020716
    # A0002M               1        1  1.547669  ...  0.387643  0.988567  1.081877

    df['CohortType'] = 'NEU'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')
    ## as discussed, visit 2 only
    df = df[df['TimePeriod'] == 2]
    ## as discussed, no fish in past 48hrs for v2
    #df = df[df['fish_pu_v2'] == 0]
    ## predict dilution for visit 2
    dilution = predict_dilution(df, 'NEU')
    
    
    dilution['PIN_Patient'] = dilution['PIN_Patient'].astype(int).astype(str)
    df_new = df.merge(dilution, on = 'PIN_Patient', how = 'left')
    
    # remove any sg missing 
    df_new = df_new[~df_new['SPECIFICGRAVITY_V2_x'].isna()]
    return df_new

def get_dataframe_nofish():
    """Returns a pandas DataFrame with fish removed for cohort"""

    df = get_dataframe()

    neu_logic = (df['fish_pu_v2'] == 0) & (df['fish'] == 0)

    df_nofish = df[neu_logic]

    return df_nofish

def get_dataframe_BLOD():
    """Returns a pandas DataFrame"""

    # First is necessary to pivot the raw NEU dataset so it matches
    # the requested features.

    # This queries the RawNEU dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawNEU.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    ## birth weight and length
  
    df['birthWt'] = df['birthWt'] * 1000
    df['birthLen'] = df['birthLen'] * 2.54

    ## education connversion

    df['ed'] = df['ed'].astype(float)

    conditions = [
    (df['ed'] <= 8 ),
    (df['ed'].isin([9,10,11,12])),
    (df['ed'].isin([13,14])) ,
    (df['ed'] == 15 ),
    (df['ed'] > 15 ) &  (df['ed'] < 20) ,
    ]
    choices = [1,2,3,4,5]

    df['education'] = np.select(conditions, choices, default=-9)
    ## birth year
    df['PPDATEDEL'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce')
    df['birth_year'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce').dt.year
    
    ## new covariates

    ## df.rename(columns = {'pregnum':'parity'}, inplace = True)
    #new covars

    covars = ['Outcome_weeks', 'age', 'ethnicity', 'race', 
    'BMI', 'smoking', 'parity', 'preg_complications',
    'folic_acid_supp', 'fish', 'babySex', 'birthWt', 'birthLen', 'headCirc',
    'WeightCentile','LGA','SGA','ga_collection','education', 'birth_year', 
    'SPECIFICGRAVITY_V2', 'fish_pu_v2']

    #calculate extra variables
    #parity
    df['parity'] = df['pregnum']
    #ga at collection

    # Pivoting the table and reseting index
    numerical_values = 'BLOD2'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome']
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod', 'Outcome'] + covars

    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns,
                        aggfunc=np.average)
                        
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

    df['CohortType'] = 'NEU'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')
    
    return df


    def get_dataframe_orig():
    """Returns a pandas DataFrame"""

    # First is necessary to pivot the raw NEU dataset so it matches
    # the requested features.

    # This queries the RawNEU dataset and excludes some of the values
    # TODO - Should we drop NaN here?
    df = pd.DataFrame.from_records(
        RawNEU.objects.
        # exclude(Creat_Corr_Result__lt=-1000).
        # exclude(Creat_Corr_Result__isnull=True).
        values()
    )

    ## birth weight and length
  
    df['birthWt'] = df['birthWt'] * 1000
    df['birthLen'] = df['birthLen'] * 2.54

    ## education connversion

    df['ed'] = df['ed'].astype(float)

    conditions = [
    (df['ed'] <= 8 ),
    (df['ed'].isin([9,10,11,12])),
    (df['ed'].isin([13,14])) ,
    (df['ed'] == 15 ),
    (df['ed'] > 15 ) &  (df['ed'] < 20) ,
    ]
    choices = [1,2,3,4,5]

    df['education'] = np.select(conditions, choices, default=-9)
    ## birth year
    df['PPDATEDEL'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce')
    df['birth_year'] = pd.to_datetime(df['PPDATEDEL'],errors='coerce').dt.year
    
    ## new covariates

    ## df.rename(columns = {'pregnum':'parity'}, inplace = True)
    #new covars

    covars = ['Outcome_weeks', 'age', 'ethnicity', 'race', 
    'BMI', 'smoking', 'parity', 'preg_complications',
    'folic_acid_supp', 'fish', 'babySex', 'birthWt', 'birthLen', 'headCirc',
    'WeightCentile','LGA','SGA','ga_collection','education', 'birth_year', 
    'SPECIFICGRAVITY_V2', 'fish_pu_v2']

    #calculate extra variables
    #parity
    df['parity'] = df['pregnum']
    #ga at collection

    # Pivoting the table and reseting index
    numerical_values = 'Result'

    columns_to_indexes = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome', 'LOD']
    categorical_to_columns = ['Analyte']
    indexes_to_columns = ['PIN_Patient','Member_c', 'TimePeriod', 'Outcome'] + covars

    df = pd.pivot_table(df, values=numerical_values,
                        index=columns_to_indexes,
                        columns=categorical_to_columns,
                        aggfunc=np.average)
                        
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

    df['CohortType'] = 'NEU'
    df['TimePeriod'] = pd.to_numeric(df['TimePeriod'], errors='coerce')
    
    return df


