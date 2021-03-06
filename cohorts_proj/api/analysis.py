import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import traceback
import statsmodels.api as sm


from datasets.models import RawFlower, RawUNM, RawDAR
from django.contrib.auth.models import User

from api import adapters

def getCorrelationPerVisit(data, x_cols, y_cols, corr_method):
    'returnn correlationns for sets of features per time period / visit'
    
    for col in [x_cols] + [y_cols]:
        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

        df1 = data
        rez = []
        seen = []
        N = None
        
        for x in x_cols:
            for y in y_cols:
                
                if x!=y:

                    for visit in df1['TimePeriod'].unique():
                        #print(visit)
                        df_visit = df1[df1['TimePeriod']== visit]
                        #print(df1.shape)
                        try:
                            temp = df_visit[(~df_visit[x].isna()) & (~df_visit[y].isna()) ]
                            N = temp.shape[0]
                            if corr_method == 'spearman':
                                spearman = stats.spearmanr(temp[x], temp[y])
                                rez.append([x,y,N,visit,spearman.correlation,spearman.pvalue])
                            else:
                                spearman = stats.pearsonr(temp[x], temp[y])
                                rez.append([x,y,N,visit,spearman[0],spearman[1]])

                        except:
                            print('err')  
                            
    return pd.DataFrame(rez, columns = ['x','y','N','visit','corr','pval']).sort_values(by = 'pval') 


def getCorrelation(data, x_cols, y_cols, corr_method):
    
    for col in [x_cols] + [y_cols]:
        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

        df1 = data
        rez = []
        seen = []
        N = None
        
        for x in x_cols:
            for y in y_cols:
                
                if x!=y:
                    try:
                        temp = df1[(~df1[x].isna()) & (~df1[y].isna())]
                        N = temp.shape[0]
                        if corr_method == 'spearman':
                            spearman = stats.spearmanr(temp[x], temp[y])
                            rez.append([x,y,N,spearman.correlation,spearman.pvalue])
                        else:
                            spearman = stats.pearsonr(temp[x], temp[y])
                            rez.append([x,y,N,spearman[0],spearman[1]])
                        
                        
                    except:
                        print('err')      

    return pd.DataFrame(rez, columns = ['x','y','N','corr','pval']).sort_values(by = 'pval')

def getCorrelationHeatmap(data, to_corr_cols):

    for col in to_corr_cols:

        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

    sns.set_theme(style="white",font_scale=1.75)

    # Compute the correlation matrix
    corr = data[to_corr_cols].corr(method = 'spearman').round(2)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40, 30))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(0, 230, as_cmap=True)

    g = sns.heatmap(corr, mask=mask,
                cmap = cmap, vmax=.3, center=0, annot = True, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    #g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 16)
 

    # Draw the heatmap with the mask and correct aspect ratio
    return g

def cohortdescriptive(df_all):
    'fuction that returns count, mean, and std per cohort'

    b = df_all.groupby(['CohortType']).agg(['count','mean','std']).transpose().reset_index()

    df2 = b.pivot(index='level_0', columns='level_1', values=['NEU','DAR','UNM'])

    df2.columns = list(map("_".join, [[str(x[0]),x[1]] for x in list(df2.columns)]))

    return df2

def cohortdescriptiveOverall(data):

    for col in data.columns:

        try:
            data[col] = data[col].astype(float)
            
        except:
            data[col] = data[col]

    df_all = data

    cohort = df_all['CohortType'].unique()[0]

    b = df_all.groupby(['CohortType']).agg(['count','mean','std']).transpose().reset_index()

    df2 = b.pivot(index='level_0', columns='level_1', values=[cohort])

    df2.columns = list(map("_".join, [[str(x[0]),x[1]] for x in list(df2.columns)]))

    return df2

def cohortDescriptiveByOutcome(data):

    for col in data.columns:

        try:
            data[col] = data[col].astype(float)
        
        except:
            data[col] = data[col]

    b = data.groupby(['Outcome']).agg(['count','mean','std']).transpose().reset_index()

    df2 = b.pivot(index='level_0', columns='level_1', values=[0.0,1.0])

    df2.columns = list(map("_".join, [[str(x[0]),x[1]] for x in list(df2.columns)]))

    return df2

def oneHotEncoding(df, toencode):

    #TODO: add onehot encoding for race, gender, etc.

    for var in toencode:
        dum = pd.get_dummies(df[var], prefix=var)

    return dum

def merge3CohortFrames(df1,df2,df3):
    'merge on feature intersections'

    for as_feature in ['UASB', 'UDMA', 'UAS5', 'UIAS', 'UAS3', 'UMMA']:
        if as_feature not in df1.columns:
            df1[as_feature] = np.nan
        if as_feature not in df2.columns:
            df2[as_feature] = np.nan
        if as_feature not in df3.columns:
            df3[as_feature] = np.nan



    s1 = set(df1.columns)
    s2 = set(df2.columns)
    s3 = set(df3.columns)

    cc = set.intersection(s1, s2, s3)

    df_all = pd.concat([df1[cc],df2[cc],df3[cc]])

    for x in df_all:
        try:
            df_all[x] = df_all[x].astype(float)
        except:
            pass

    return df_all

def merge2CohortFrames(df1,df2):
    'merge on feature intersections'

    s1 = set(df1.columns)
    s2 = set(df2.columns)
    cc = set.intersection(s1, s2)

    df_all = pd.concat([df1[cc],df2[cc]])

    for x in df_all:
        try:
            df_all[x] = df_all[x].astype(float)
        except:
            pass

    return df_all

def numberParticipants(df_all):

    df_all

def categoricalCounts(df,categorical, indv_or_all = 1):

    df22 = df[categorical].drop_duplicates(['PIN_Patient'])

    categorical.remove('PIN_Patient')

    df22 = df22[categorical]
    ## for all cohorts
    if indv_or_all == 1:
        df33 = pd.DataFrame(pd.melt(df22,id_vars=['CohortType'])\
                        .groupby(['Analyte','value'])['value'].count())
        

        df33.index.names = ['variable', 'cat']
    ## counts by cohort individually
    if indv_or_all == 0:
        df33 = pd.DataFrame(pd.melt(df22,id_vars=['CohortType'])\
                        .groupby(['Analyte','value'])['value'].count())
        

        df33.index.names = ['CohortType','variable', 'cat']

    return df33.reset_index()

def linearreg(df, covars, target):

    Y = df[target]
    X = df[covars]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    
    return results

def linearreg(df, x_vars, targets, cohort):

    rez = []

    df = df.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)


    for period in df['TimePeriod'].unique():
        df_period = df[df['TimePeriod'] == period]

        for x in x_vars:
            for y in targets:
                try:
                    df_temp = df_period[(~df_period[x].isna()) & (~df_period[y].isna()) ]
                    print('*************')
                    print(df_temp.shape)

                    X = df_temp[x]
                    Y = df_temp[y]
                    
                    res = stats.linregress(X, Y)
                    print(res)
                    rez.append([cohort, x, y, len(X),res.slope, res.intercept, 
                                            res.rvalue, res.pvalue, res.stderr,res.intercept_stderr])
                    #except Exception as e:
                    #    print(e)
                        
                    #    rez.append([cohort, x, y, len(X), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                except Exception as e:
                    print(e)

    rez_df = pd.DataFrame(rez, columns = ['cohort','x','y', 'N','slope', 'intercept','rvalue', 'pvalue', 'stderr','intercept_stderr'])

    return rez_df





def runcustomanalysis():

    ## Get data
    df1 = adapters.neu.get_dataframe()
    df2 = adapters.unm.get_dataframe()
    df3 = adapters.dar.get_dataframe()


    ## should be presennt in both DAR and UNM
    for as_feature in ['UASB', 'UDMA', 'UAS5', 'UIAS', 'UAS3', 'UMMA']:
        if as_feature not in df1.columns:
            df1[as_feature] = np.nan
        if as_feature not in df2.columns:
            df2[as_feature] = np.nan
        if as_feature not in df3.columns:
            df3[as_feature] = np.nan

    df_merged2 = merge2CohortFrames(df2,df3)

    df_merged2.columns

    ## merge the frames

    df_merged = merge3CohortFrames(df1,df2,df3)

    ## distiinguish features between covariates and analytes

    covariates = ['PIN_Patient', 'TimePeriod', 'Member_c', 'Outcome', 'Outcome_weeks',
                'age', 'ethnicity', 'race', 'BMI', 'smoking', 'parity',
                    'preg_complications', 'folic_acid_supp', 
                    'fish',
                    'babySex',
                    'birthWt',
                    'birthLen']

    analytes =    ['UBA', 'UBE', 'UCD', 'UCO', 'UCR', 'UCS', 'UCU', 'UHG',
                'UMN', 'UMO', 'UNI', 'UPB', 'UPT', 'USB', 'USE', 'USN', 'UTAS', 'UTL',
                'UTU', 'UUR', 'UVA', 'UZN']

    analytes_arsenic_neu = ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 

    analytes_arsenic_dar = ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 

    analytes_arsenic_unm = ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 

    continuous = ['Outcome_weeks','age','BMI','fish','birthWt','birthLen','WeightCentile'] + analytes_arsenic_neu 

    categorical = ['CohortType','TimePeriod','Member_c','Outcome','folic_acid_supp',
                'ethnicity','race','smoking','preg_complications','babySex','LGA','SGA']


    ## Number of Participants

    output_path = 'mediafiles/analysisresults/'
    #output_path = '../mediafiles/analysisresults/'

    print('Files written to:')
    print(output_path)

    df_merged.groupby(['CohortType'])['PIN_Patient'].nunique()\
            .to_csv(output_path + 'number_unique_participats.csv', index = True)

    ## Number of Records

    df_merged.groupby(['CohortType'])['PIN_Patient'].count()\
            .to_csv(output_path + 'number_unique_records.csv', index = True)
    

    ## Number of participants per visit

    par_per_vis = df_merged.groupby(['CohortType','TimePeriod'])['PIN_Patient'].nunique().reset_index()
    par_per_vis.to_csv(output_path +'number_unique_participats_per_visit.csv', index = True)

    #################################################################################################################################### 


    ## Continous descriptions

    cohortdescriptive(df_merged[['CohortType'] + continuous]).reset_index()\
        .to_csv(output_path + 'continous_merged_descriptive_by_outcome.csv')

    ####################################################################################################################################

    ## Describe by outcome

    cohortDescriptiveByOutcome(df_merged[continuous + ['Outcome']]).reset_index()\
        .to_csv(output_path + 'continous_merged_descriptive_by_outcome.csv')

    ####################################################################################################################################

    ## Decribe categorical variables and percentages

    df1_cat = categoricalCounts(df_merged,categorical + ['PIN_Patient'], 1).reset_index()

    df1_tos = categoricalCounts(df_merged,categorical + ['PIN_Patient'], 1).groupby(['variable']).sum().reset_index()

    df22 = df1_cat.merge(df1_tos, on = 'variable')

    df22['percent'] = df22['value_x'] / df22['value_y']
    df22.round(4)\
        .to_csv(output_path + 'categorical_counts_and_percentage.csv', index = False)

    to_corr_cols = continuous

    ##correlatiom heatmap

    corr1 = getCorrelationHeatmap(df_merged, continuous)

    plt.savefig(output_path + '_combiend_correlations.png')

    ##correlatiom heatmap

    for cohort in df_merged['CohortType'].unique():
        df_cohort = df_merged[df_merged['CohortType'] == cohort]

        for period in df_cohort['TimePeriod'].unique():

            df_indv = df_cohort[df_cohort['CohortType']== cohort]
            
            corr_i = getCorrelationHeatmap(df_indv, continuous)

            plt.savefig(output_path + cohort + str(period) + '_indiv_correlations.png')
    
    ## spearma correlations written to file

    x_cols =  [ 'age',
            'BMI', 'parity',
            'folic_acid_supp', 'fish', 'babySex', 'birthWt', 'birthLen'] + analytes_arsenic_neu
    
    y_cols = ['Outcome_weeks','birthWt','birthLen']

    for cohort in df_merged['CohortType'].unique():
        df_cohort = df_merged[df_merged['CohortType'] == cohort]

        for period in df_cohort['TimePeriod'].unique():
            
            df_period = df_cohort[df_cohort['TimePeriod'] == period]
            
            corr_period = getCorrelation(df_period, x_cols, y_cols,'spearman').round(4)
            
            corr_period['TimePeriod'] = period

            corr_period['Cohort'] = cohort
            
            corr_period.round(4).to_csv(output_path + str(cohort) + str(period)  + '_period_correlations.csv', index = False)
    
    ##correlatiom heatmap
    
    x_cols =  ['folic_acid_supp', 'fish']

    y_cols = analytes_arsenic_neu

    #getCorrelation(df3, x_cols, y_cols,'spearman').round(4).to_csv(output_path + 'dar_1_correlations.csv')

    x_cols = ['PNFFQFR_FISH_KIDS','PNFFQSHRIMP_CKD','PNFFQDK_FISH','PNFFQOTH_FISH','mfsp_6','fish','TOTALFISH_SERV']
    y_cols = ['UIAS', 'UASB', 'UAS3', 'UAS5','UHG','UAS']

    #getCorrelation(df3, x_cols, y_cols,'spearman').round(4).to_csv(output_path + 'dar_fish_correlations.csv')

    ###

    x_cols = ['UIAS', 'UASB', 'UAS3', 'UAS5','UHG','UAS','UTAS']
    y_cols = ['Outcome_weeks']

    #getCorrelation(df3, x_cols, y_cols,'spearman').round(4).to_csv(output_path + 'dar_3_correlations.csv')

    ### univariate linear regression:

    confound = continuous + x_cols

    neu = linearreg(df1, confound, ['Outcome_weeks','birthWt'], 'NEU')

    unm = linearreg(df2, confound, ['Outcome_weeks','birthWt'], 'UNM')

    dar = linearreg(df3, confound, ['Outcome_weeks','birthWt'], 'DAR')

    pd.concat([neu,unm,dar]).round(4).to_csv(output_path +  'liregress_results.csv', index = False)