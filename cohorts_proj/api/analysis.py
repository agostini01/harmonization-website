import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

import traceback


def getSpearmans(data, x_cols, y_cols):
    
    for col in [x_cols] + [y_cols]:

        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

        df1 = data

        rez = []
        seen = []

        for x in x_cols:
            for y in y_cols:
                
                if x!=y:
                    
                    try:
                        temp = df1[(~df1[x].isna()) & (~df1[y].isna())]
                        
                        spearman = stats.spearmanr(temp[x], temp[y])
                        
                        rez.append([x,y,spearman.correlation,spearman.pvalue])
                    except:
                        
                        print('err')      

    return pd.DataFrame(rez, columns = ['x','y','corr','pval']).sort_values(by = 'pval')


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

    g = sns.heatmap(corr, mask=mask, cmap = cmap, vmax=.3, center=0, annot = True, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    #g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 16)
 

    # Draw the heatmap with the mask and correct aspect ratio
    return g

def cohortdescriptive(df1,df2,df3):

    s1 = set(df1.columns)
    s2 = set(df2.columns)
    s3 = set(df3.columns)

    cc = set.intersection(s1, s2, s3)

    curr_harm = pd.concat([df1[cc],df2[cc],df3[cc]]).reset_index()

    for x in curr_harm.columns:
        try:
            curr_harm[x] = curr_harm[x].astype(float)
            curr_harm.loc[curr_harm[x] < 0, x] = np.nan
        except:
            curr_harm[x] = curr_harm[x]

    count = curr_harm.groupby(['CohortType']).agg(['count']).transpose().reset_index()
    mean = curr_harm.groupby(['CohortType']).agg(['mean']).transpose().reset_index()
    std = curr_harm.groupby(['CohortType']).agg(['std']).transpose().reset_index()

    count.drop(['level_1'], axis = 1, inplace = True)
    mean.drop(['level_1'], axis = 1, inplace = True)
    std.drop(['level_1'], axis = 1, inplace = True)

    count.columns = [x + '_cnt' for x in count.columns]
    mean.columns = [x + '_mean' for x in mean.columns]
    std.columns = [x + '_std' for x in std.columns]

    rez = count.merge(mean, left_on = 'level_0_cnt', right_on = 'level_0_mean').drop(['level_0_mean'],axis = 1)

    return rez
