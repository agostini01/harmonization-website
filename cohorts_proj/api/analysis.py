import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import os


import traceback

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

    df2.to_csv('test.csv')
    
    return df2

def oneHotEncoding(df, toencode):

    #TODO: add onehot encoding for race, gender, etc.

    for var in toencode:
        dum = pd.get_dummies(df[var], prefix=var)

    return dum

def merge3CohortFrames(df1,df2,df3):
    'merge on feature intersections'

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

def categoricalCounts(df,categorical):

    df22 = df[categorical].drop_duplicates(['PIN_Patient'])

    categorical.remove('PIN_Patient')

    df22 = df22[categorical]

    df33 = pd.DataFrame(pd.melt(df22,id_vars=['CohortType'])\
                    .groupby(['CohortType','Analyte','value'])['value'].count())

    df33.index.names = ['CohortType', 'variable', 'cat']

    df33.reset_index().head()

    return df33



