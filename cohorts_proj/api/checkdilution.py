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

from api import analysis
from api import adapters

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
    df[['UTAS','UTAS_adj']].describe().round(4).to_csv(path + "Original and ADjusted concentrations {}.csv".format(cohort), index = True)

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