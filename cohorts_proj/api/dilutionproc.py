import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import traceback
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels
import bambi as bmb
import arviz as az

from datasets.models import RawFlower, RawUNM, RawDAR
from django.contrib.auth.models import User

#from api.analysis import add_confound
from api import adapters

#dilution prediction procedure


#Jordan's instructions are:

#1) calculated specific gravity (or creatinine) Z-scores, 
#2) ran regression models in which the specific gravity (or creatinine) Z-score is the outcome and predictors are 
#variables that are associated with the urinary dilution proxy as well as exposure and outcome of interest, 
#3) predicted the specific gravity or creatinine Z-scores from these regression models, 
#4) back-transformed the Z-scores into the original units, 
##5) calculated dilution ratios for each person by taking their predicted specific gravity (or creatinine) value and dividing it by their observed value, 
#6) multiplied the person’s observed phthalate concentration by their dilution ratio, 
#7) in final regression models utilizing this dilution-corrected biomarker you also include an indicator variable for whether specific gravity 
#or creatinine was used.

def predict_dilution(df_merged, cohort):
    'calculate predicted dilutions per cohort (UNM - Predcreatinine, DAR/NEU - PredSG)'

    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    df_merged = df_merged.drop_duplicates(['PIN_Patient'], keep = 'first')
    #select the cohort
    df_merged = df_merged[df_merged['CohortType'] == cohort]
    #covariates needed for prediction
    #Where covariates = race + education + age + pre-pregnancy BMI + 
    #                   gestational age at sample collection + year of delivery + study site
    #these have to be in the dataset
    dilution_covars = ['race', 'education','babySex','BMI', 'ga_collection','birth_year']
    x_feature = 'age'

    #1) calculated specific gravity (or creatinine) Z-scores, <br>
    originalSize = df_merged.shape[0]
    
    if cohort == 'NEU':
        orig_dilution = 'SPECIFICGRAVITY_V2'
        mean = df_merged['SPECIFICGRAVITY_V2'].mean()
        std =  df_merged['SPECIFICGRAVITY_V2'].std()
        
        df_merged['zscore'] = (df_merged['SPECIFICGRAVITY_V2'] -  mean) / std                                                     
        y_feature = 'zscore'
    
    if cohort == 'DAR':
        orig_dilution = 'urine_specific_gravity'
        mean = df_merged['urine_specific_gravity'].mean()
        std =  df_merged['urine_specific_gravity'].std()
        
        df_merged['zscore'] = (df_merged['urine_specific_gravity'] -  mean) / std                                                        
        y_feature = 'zscore'
        
    if cohort == 'UNM':
        orig_dilution = 'creatininemgdl'
        mean = df_merged['creatininemgdl'].mean()
        std =  df_merged['creatininemgdl'].std()
        
        df_merged['zscore'] = (df_merged['creatininemgdl'] -  mean) / std                                                        
        y_feature = 'zscore'
    
    # convert the categorical variables to 0-1 dummy/ one-hot encoding
    data = df_merged

    sizeAfterConfound = data.shape[0]
    #assert originalSize == sizeAfterConfound
    
    data.drop(['CohortType'], inplace = True, axis = 1)
    data['intercept'] = 1
    ids = data['PIN_Patient'].values
    data = data.select_dtypes(include = ['float','integer'])
    
    X = data[[x for x in data.columns if x !=y_feature and x!= 'PIN_Patient' and x!= orig_dilution]]
    Y = data[y_feature]

    # Fit the linear regression 
    if X.shape[0] > 1:
        reg = sm.OLS(Y, X).fit() 
        ret = reg.summary()
    else:
        ret = 'no samples'
        assert 1 == 0

    # predict z-scores
    Y_pred = reg.predict(X)
    
    #Y = original
    #Y_pred = prediction
    #X = dataset
    #ids = patient id
    #ret = summary from regression model.
    
    #zip predicted values with original and ids
    df_out = pd.DataFrame(list(zip(ids, Y, Y_pred)), columns = ['PIN_Patient','original', 'prediction'])
    assert df_out.shape[0] <= originalSize

    data['PIN_Patient'] = ids

    #merge original data with predictions
    check_ids = data.merge(df_out, on = 'PIN_Patient')
    print('Model out {}. afterocnf {}. check ids {}'.format(df_out.shape[0], sizeAfterConfound, check_ids.shape[0]))

    # check size
    assert check_ids.shape[0] == sizeAfterConfound
    # check idx
    assert check_ids[y_feature].values.tolist() == check_ids['original'].values.tolist()
    
    #4) back-transformed the Z-scores into the original units, 
    df_out['prediction_xvalue'] = df_out['prediction'] * std + mean
    df_out['original_xvalue'] = df_out['original'] * std + mean
    
    df_out2 = df_out.merge(df_merged[['PIN_Patient',orig_dilution, 'zscore']], on = 'PIN_Patient')

    #5) calculated dilution ratios for each person by taking their predicted specific gravity (or creatinine) value and dividing it by their observed value
    # UDR for specific gravity dilution
    if cohort in ['NEU', 'DAR']:
        df_out2['UDR'] = (df_out2['original_xvalue']-1) / (df_out2['prediction_xvalue'] - 1)
        dil_indicator = 0
    # UDR for creatinine urine dilution
    if cohort == 'UNM':
        df_out2['UDR'] = df_out2['original_xvalue'] / df_out2['prediction_xvalue']
        dil_indicator = 1
    
    df_out2['Cohort'] = cohort
    df_out2['dil_indicator'] = dil_indicator
    
    return df_out2

#function that will print out statistics and figures about actual and predited sg z scores.
def OriginalVsAdjustedZscore(df, path):

    cohort = df['CohortType'].values[0]

    #stats
    df[['original','prediction']].describe().round(4).to_csv(path + "Original and Predicted Specific Gravity zscore {}.csv".format(cohort), index = True)
    
    #figure
    x = df['original']
    y = df['prediction']

    plt.figure(figsize=(14, 7))

    plt.hist(x, bins=50, color='c', alpha=.55, label = 'Original SG')

    plt.axvline(x.median(), color='k', linestyle='dashed', linewidth=1.5)
    plt.text(x.median(), 30,'Orig = ' + str(round(x.median(),4)), size = 15)

    plt.hist(y, bins=50, color='r',  alpha=0.3, label = 'Predicted SG')

    plt.axvline(y.median(), color='k', linestyle='dashed', linewidth=1.5)
    plt.text(y.median() + -.75, 20, 'Pred = ' + str(round(y.median(),4)), size = 15)

    plt.xlabel("sg zscore", size=14)
    plt.ylabel("Count", size=14)
    #plt.title("Original and Predicted Specific Gravity dilution")
    plt.title("Original and Adjusted Arsenic")

    plt.legend(loc='upper right')
    plt.savefig(path + "Original and Predicted Specific Gravity zscore {}.png".format(cohort))

#function that will print out stats/fig for adjusted and unadjusted concentrations
def OriginalVsAdjustedAnalyte(df, path):

    cohort = df['CohortType'].values[0]

    # table stats
    df['UTAS_adj'] = df['UTAS'] / df['UDR']
    df[['UTAS','UTAS_adj']].describe().round(4).to_csv(path + "Original and Adjusted concentrations {}.csv".format(cohort), index = True)

    # figure
    x = np.log(df['UTAS'])
    y = np.log(df['UTAS'] / df['UDR'])

    plt.figure(figsize=(14, 7))

    plt.hist(x, bins=50, color='c', alpha=0.50, label = 'Original')

    plt.axvline(x.median(), color='k', linestyle='dashed', linewidth=1.5)
    plt.text(x.median()-2, 35, 'Median Original = ' + str(round(x.median(),4)), size = 15)

    plt.hist(y, bins=50, color='r',  alpha=0.50, label = 'Adjusted')

    plt.axvline(y.median(), color='k', linestyle='dashed', linewidth=1.5)
    plt.text(y.median()+.05, 20, 'Median Adjusted = ' + str(round(y.median(),4)), size = 15)

    plt.xlabel("Concentration log(UTAS)", size=14)
    plt.ylabel("Count", size=14)
    plt.title("Original Concentration vs. Adjusted")
    plt.legend(loc='upper right')
    plt.savefig(path + "Original UTAS Concentration vs.Adjusted by method_{}.png".format(cohort))


def generatedilutionstats():

    ## Model 1: Restricted to participants with no fish/seafood consumption.

    ## Get data frames
    df_NEU = adapters.neu.get_dataframe()
    df_UNM = adapters.unm.get_dataframe()
    ##df_DAR = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

    frames_for_analysis = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR)
    ]
    for name, df in frames_for_analysis:
        print('Data Stats')
        print(name)
        print(df.shape)
    
    # set output paths for results:

    dcheck_path = '/usr/src/app/mediafiles/analysisresults/dilutioncheck/'
    
    try:
        os.mkdir(dcheck_path)
        os.mkdir(dcheck_path)
    except:
        print('Exists')

    for name, df in frames_for_analysis:
        
        OriginalVsAdjustedZscore(df, dcheck_path)
        OriginalVsAdjustedAnalyte(df, dcheck_path)