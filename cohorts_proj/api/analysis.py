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
                        
                        df_visit = df1[df1['TimePeriod']== visit]
                        
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

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            df_temp = df[(~df[col].isna()) & (~df[col2].isna())]
            if df_temp.shape[0] > 2:
                spearman = stats.spearmanr(df_temp[col], df_temp[col2])
                p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = spearman.pvalue
            else:
                p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = 1


    return p_matrix

def getCorrelationHeatmap(data, to_corr_cols):

    for col in to_corr_cols:

        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

    #sns.set_theme(style="white",font_scale=1.75)

    # Compute the correlation matrix
    corr = data[to_corr_cols].corr(method = 'spearman').round(4)

    # Generate a mask for the upper triangle
    p_values = corr_sig(data[to_corr_cols])
    mask = np.invert(np.tril(p_values<0.05))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40, 30))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(0, 230, as_cmap=True)

    g = sns.heatmap(corr, 
                cmap = cmap, vmax=.3, center=0, annot = True, 
                square=True, linewidths=.5, annot_kws={"size": 35}, mask=mask)

    #g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 40)

    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 30, rotation = 90)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 30, rotation = 0)

    # Draw the heatmap with the mask and correct aspect ratio
    return g

def cohortdescriptive(df_all):
    'fuction that returns count, mean, and std per cohort'

    df_all = df_all.drop_duplicates(['CohortType','PIN_Patient','TimePeriod'])

    b = df_all.groupby(['CohortType']).agg(['count','mean','std']).transpose().reset_index()

    df2 = b.pivot(index='level_0', columns='level_1', values=['NEU','DAR','UNM'])

    df2.columns = list(map("_".join, [[str(x[0]),x[1]] for x in list(df2.columns)]))

    return df2

def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.median()

def q3(x):
    return x.quantile(0.75)

def cohortdescriptive_all(df_all):
    ' summary; minimum, quartile 1, median, quartile 3, and maximum.'
    
    df_all = df_all.drop_duplicates(['CohortType','PIN_Patient','TimePeriod'])

    df_all = df_all.select_dtypes(include=['float64'])

    categorical = ['CohortType','TimePeriod','Member_c','Outcome','folic_acid_supp', 'PIN_Patient',
                'ethnicity','race','smoking','preg_complications','babySex','LGA','SGA','education']

    df_all = df_all.loc[:, ~df_all.columns.isin(categorical)]

    #b = df_all.agg(['count','mean','std',lambda x: x.quantile(0.25), lambda x: x.quantile(0.50)])

    df_all[df_all < 0 ] = np.nan

    b = df_all.agg(['count','mean','std','min',  q1, 'median', q3, 'max']).transpose().round(4)


    return b

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
    #only consider visit 2 for NEU
    df2 = df2[df2['TimePeriod'] == 2]

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

    for as_feature in ['UASB', 'UDMA', 'UAS5', 'UIAS', 'UAS3', 'UMMA']:
        if as_feature not in df1.columns:
            df1[as_feature] = np.nan
        if as_feature not in df2.columns:
            df2[as_feature] = np.nan
    
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

def categoricalCounts(df):

    #each participant should only have 1 measurment per fvariable

    categorical = ['CohortType','TimePeriod','Member_c','Outcome','folic_acid_supp', 'PIN_Patient',
                'ethnicity','race','smoking','preg_complications','babySex','LGA','SGA','education']
        
    df22 = df[categorical].drop_duplicates(['PIN_Patient'])

    categorical.remove('PIN_Patient')

    df22 = df22[categorical]
    
    df33 = pd.DataFrame(pd.melt(df22,id_vars=['CohortType'])\
                        .groupby(['Analyte','value'])['value'].count())
        
    df33.index.names = ['variable', 'cat']
   
    return df33.reset_index()

def linearreg(df, x_vars, targets, cohort):
    # run simple linear regression y = ax + b and report slope, intercept, rvalue, plvalue, 'stderr

    rez = []

    df = df.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    for period in df['TimePeriod'].unique():
        df_period = df[df['TimePeriod'] == period]

        for x in x_vars:
            for y in targets:
                try:
                    df_temp = df_period[(~df_period[x].isna()) & (~df_period[y].isna())  & (df_period[y] >= 0) & (df_period[x] >= 0) ]

                    X = df_temp[x]
                    Y = df_temp[y]
                    
                    res = stats.linregress(X, Y)

                    rez.append([cohort, x, y, len(X),res.slope, res.intercept, 
                                            res.rvalue, res.pvalue, res.stderr,res.intercept_stderr])

                except Exception as e:
                    print(e)

    rez_df = pd.DataFrame(rez, columns = ['cohort','x','y', 'N','slope', 'intercept','rvalue', 'pvalue', 'stderr','intercept_stderr'])

    return rez_df

def logisticregress(df, x_vars, targets, cohort):
    # run simple logistic regression log(p(x)/1-p(x)) = ax + b and report slope, intercept, rvalue, plvalue, 'stderr

    rez = []

    df = df.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    for period in df['TimePeriod'].unique():
        df_period = df[df['TimePeriod'] == period]

        for x in x_vars:
            for y in targets:
                try:
                    df_temp = df_period[(~df_period[x].isna()) & (~df_period[y].isna()) & (~df_period[y].isin([0.0,1.0,0,1]))]
            
                    df_temp['intercept'] = 1
                    X = df_temp[['intercept',x]]
                    df_temp[y] = df_temp[y].astype(int)
                    Y = df_temp[y]

                    #res = stats.linregress(X, Y)
                    if df_temp.shape[0] > 2:
                        log_reg = sm.Logit(Y, X).fit() 

                        intercept = log_reg.params[0]
                        coef = log_reg.params[1]

                        intercept_p = log_reg.pvalues[0]
                        coef_p = log_reg.pvalues[1]

                        rez.append([cohort, x, y, len(X),coef, coef_p, intercept, intercept_p])
                        
                    #    rez.append([cohort, x, y, len(X), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                except Exception as e:
                    print(e)

    rez_df = pd.DataFrame(rez, columns = ['cohort','x','y', 'N','slope', 'slope_p', 'intercept','intercept_p'])

    return rez_df


def crude_reg(df_merged, x_feature, y_feature, covars, adjust_dilution, output):
    # inro for crude simple regression y = ax + b and report full results
    # y_feature has to be binary (i.e. 0,1)
    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    df_merged = df_merged[(~df_merged[x_feature].isna()) & (~df_merged[y_feature].isna())]
    #make sure all concentrations are above 0 - assuption is ok because lowest conc should have LOD
    df_merged = df_merged[(df_merged[x_feature]> 0) & (~df_merged[x_feature].isna())  ]
    split_covars = covars.split('|')

    ## adjust dilution
    if adjust_dilution == 'True':
        df_merged[x_feature] = df_merged[x_feature] / df_merged['UDR']

    if len(split_covars) > 0:
        data = add_confound(df_merged, x_feature, y_feature, split_covars)
    else:
        data = df_merged

    data = data[(data[x_feature]> 0) & (~data[x_feature].isna())  ]

    data_copy = data.copy()
    data.drop(['CohortType'], inplace = True, axis = 1)

    data['intercept'] = 1
    #TODO: clean up
    try:
        data['babySex'] = data['babySex'].astype(float)
    except:
        xsd = 1
    try:
        data['parity'] = data['parity'].astype(float)
    except:
        xsd = 1
  
    data = data.select_dtypes(include = ['float','integer'])

    #X and Y features TODO: clean up
    X = data[[x for x in data.columns if x !=y_feature and x!= 'PIN_Patient']]
    Y = data[y_feature]
    X[x_feature]= np.log(X[x_feature])

    if df_merged.shape[0] > 2:

        reg = sm.OLS(Y, X).fit() 
        ret = reg.summary()
    else:
        ret = 'error'
    # model string
    fit_string = y_feature + '~'
    
    for x in X.columns:
        if x == x_feature:
            fit_string += ' + log(' + str(x) +')'
        else:

            fit_string += ' + ' + str(x)
    
    fit_string = fit_string.replace('~ +','~')
    header = ''
    for cohort in data_copy['CohortType'].unique():
        cohort_data = data_copy[data_copy['CohortType'] == cohort]
        header += '<div> <b> ' + cohort +'  Number samples :</b> ' + str(cohort_data.shape[0]) + '</div>'
    header += '<div> <b> Total Number samples :</b> ' + str(X.shape[0]) + '</div>'
    header += '<div> <b>  Model: </b>' + fit_string + '</div>'
    header += '<div> ===================================================</div>'

    htmls = header + ret.tables[0].as_html() + ret.tables[1].as_html() 

    # depending where we are calling it from
    if output == 'csv':
        final_return = ret
    if output == 'html':
        final_return = htmls

    return final_return

def crude_logreg(df_merged, x_feature, y_feature, covars, adjust_dilution, output):
    # inro for crude simple logistic regression log(p(x)/1-p(x)) = ax + b and report slope, intercept, rvalue, plvalue, 'stderr
    # y_feature has to be binary (i.e. 0,1)
    print(df_merged.shape)
    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    print(df_merged.shape)
    df_merged = df_merged[(~df_merged[x_feature].isna()) & (~df_merged[y_feature].isna()) & \
        (df_merged[y_feature].isin([0.0,1.0,0,1, '0', '1']))]
    #make sure all concentrations are above 0 - assuption is ok because lowest conc should have LOD
    #df_merged = df_merged[df_merged[x_feature]> 0]

    print(df_merged.shape)

    #split the variables in the checkboxes
    split_covars = covars.split('|')
 
    ## adjust dilution0
    if adjust_dilution == 'True':
        df_merged[x_feature] = df_merged[x_feature] / df_merged['UDR']

    if len(split_covars) > 0:
        data = add_confound(df_merged, x_feature, y_feature, split_covars)
    else:
        data = df_merged

    #add confounding variables and adjust if they are categorical
    data = add_confound(df_merged, x_feature, y_feature, split_covars)
    data = data[(data[x_feature]> 0) & (~data[x_feature].isna())  ]

    data.drop(['CohortType'], inplace = True, axis = 1)
    # set intercept to 1
    data['intercept'] = 1

    #TODO: clean up
    try:
        data['babySex'] = data['babySex'].astype(float)
    except:
        xsd = 1
    try:
        data['parity'] = data['parity'].astype(float)
    except:
        xsd = 1

    data = data.select_dtypes(include = ['float','integer'])
    print('Data shape after intselect')
    print(data.shape)

    #independent
    X = data[[x for x in data.columns if x !=y_feature and x!= 'PIN_Patient']]
    #target
    Y = data[y_feature]
    #log of the exposure
    X[x_feature]= np.log(X[x_feature])


    if df_merged.shape[0] > 2:
        log_reg = sm.Logit(Y, X).fit()
        ret = log_reg.summary()
        
    else:
        ret = 'error'

    fit_string = y_feature + '~'
    
    for x in X.columns:
        if x == x_feature:
            fit_string += ' + log(' + str(x) +')'
        else:

            fit_string += ' + ' + str(x)
    
    fit_string = fit_string.replace('~ +',' ~')
    header = '<div> <b> Logistic Regression </b> </div>'
    header += '<div> <b> Number samples: </b> ' + str(X.shape[0]) + '</div>'
    header += '<div> <b>  Model: </b>' + fit_string + '</div>'
    header += '<div> <b> Group: </b> CohortType '
    
    htmls = header + ret.tables[0].as_html() + ret.tables[1].as_html()      

    # depending where we are calling it from
    if output == 'csv':
        final_return = ret
    if output == 'html':
        final_return = htmls

    return final_return

def crude_mixedML2(df_merged, x_feature, y_feature, covars):

    #TODO: Replace covars variable with actual selection of indivdual features

    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    split_covars = covars.split('|')

    data = add_confound(df_merged, x_feature, y_feature, split_covars)

    data['intercept'] = 1
    #data = data.select_dtypes(include = ['float','integer'])

    X = data[[x for x in data.columns if x !=y_feature and x!= 'CohortType']]

    Y = data[y_feature]

    if X.shape[0] > 2:

        reg = sm.MixedLM(Y, X, groups=data["CohortType"], exog_re=X["intercept"]).fit()
        ret = reg.summary()
    else:
        ret = 'error'

    fit_string = y_feature + '~'
    
    for x in X.columns:
        fit_string += ' + ' + str(x)
    

    fit_string = fit_string.replace('~ +','~') + ' + (1|CohortType)'
    header = '<div> <b> Liear Mixed Model with Random Intercept </b> </div>'
    header += '<div> <b> Number samples: </b> ' + str(X.shape[0]) + '</div>'
    header += '<div> <b>  Model: </b>' + fit_string + '</div>'
    header += '<div> <b> Group: </b> CohortType '
    

    htmls = header + ret.tables[0].to_html() + ret.tables[1].to_html()      
    return htmls


def crude_binomial_mixedML(df_merged, x_feature, y_feature,covars):

    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    if covars == 'False':
        
        data = df_merged[[x_feature,y_feature,'CohortType']].dropna(how = 'any', axis='rows')

        data[x_feature] = data[x_feature] + 1
        data[y_feature] = data[y_feature].astype(int)
        
        random = {"a": '0 + C(CohortType)'}

        fit_string = y_feature + '~' + x_feature

    if covars == 'True':

        random = {"a": '0 + C(CohortType)'}

        data = add_confound(df_merged, x_feature, y_feature)

            ## create the model string for 
        fit_string = y_feature + '~'

        cnt = 0
        ## filter out target, at birth, and reference dummy variables in model
        for x in data.columns:

            #data.drop(['education'], inplace = True, axis = 0)
            
            if x != 'birthWt' and x !='Outcome_weeks' and x!= 'Outcome' and x != 'PIN_Patient' and x != 'SGA' and x != 'LGA' \
                and x !='birthLen' and x != 'CohortType' and x != 'race' and x!='race_1' and x!= 'smoking' and x != 'smoking_3' \
                and x != 'education_5' and x != 'education':
                
                
                if cnt == 0:
                    fit_string += ' ' + x + ' '
                else:
                    fit_string += ' + ' + x + ' '
                cnt+=1    
        
        data[y_feature] = data[y_feature].astype(int) 

    ## miced linear model with group variable = CohortType
    md = statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula(
               fit_string, random, data)

    ##fit the model 
    mdf = md.fit_vb()


    return mdf.summary()

def crude_mixedMLbayse(df_merged, x_feature, y_feature, covars='False', logit = False):

    #TODO: Replace covars variable with actual selection of indivdual features

    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    if covars == 'False':
        
        data = df_merged[[x_feature,y_feature,'CohortType']].dropna(how = 'any', axis='rows')

        fit_string = y_feature + '~' + x_feature

    if covars == 'True':

        data = add_confound(df_merged, x_feature, y_feature)

            ## create the model string for 
        fit_string = y_feature + '~'

        cnt = 0
        ## filter out target, at birth, and reference dummy variables in model
        for x in data.columns:

            #data.drop(['education'], inplace = True, axis = 0)
            
            if x != 'birthWt' and x !='Outcome_weeks' and x!= 'Outcome' and x != 'PIN_Patient' and x != 'SGA' and x != 'LGA' \
                and x !='birthLen' and x != 'CohortType' and x != 'race' and x!='race_1' and x!= 'smoking' and x != 'smoking_3' \
                and x != 'education_5' and x != 'education':
                
                if cnt == 0:
                    fit_string += ' ' + x + ' '
                else:
                    fit_string += ' + ' + x + ' '
                cnt+=1        
    
    fit_string += '+ (1|CohortType)'
    if logit == False:
        model = bmb.Model(data)
        results = model.fit(fit_string)
    else:
        model = bmb.Model(data)
        results = model.fit(fit_string, family='bernoulli',link = 'logit')

    ## miced linear model with group variable = CohortType
    mdf = az.summary(results)
    return mdf

def add_confound(df_merged, x_feature, y_feature, conf):
    
    for col in df_merged:
        try:
            df_merged[col] = df_merged[col].astype(int)
            df_merged.loc[df_merged[col] < 0, col] = np.nan
        except:
            #print('err on ' + col)
            pp = 1

    #filter ga range
    #df_merged = df_merged[(df_merged['ga_collection'] >= 13) & (df_merged['ga_collection'] <= 28)]

    incomplete_N2 = df_merged.shape[0]

    if len(conf) > 1:

        cols_to_mix =  [x_feature, y_feature, 'PIN_Patient', 'CohortType'] + conf
    else:
        cols_to_mix = [x_feature, y_feature, 'PIN_Patient', 'CohortType']

    # drop any missing values as mixed model requires complete data
    df_nonan = df_merged[cols_to_mix].dropna(axis='rows')

    complete_N = df_nonan.shape[0]

    #df_nonan['smoking'] = df_nonan['smoking'].astype(int)
    
    ## dummy race annd smoking varible
    def add_cats(name, df_nonan, ref_val):

        df_nonan[name] = df_nonan[name].astype('float').astype(int)

        df = pd.concat([df_nonan, pd.get_dummies(df_nonan[name], prefix = name)], axis = 1)
        #print(df.columns)

        try:
            df.drop([name,name + '_' + ref_val], inplace = True, axis = 1)
        except:
            pass

        return df

    if 'race' in conf: 
        df_nonan = add_cats('race', df_nonan, '1')

    if 'smoking' in conf: 
        df_nonan = add_cats('smoking', df_nonan, '0')
    
    if 'education' in conf: 
        df_nonan = add_cats('education', df_nonan, '5')

    #dup_visit_N = df_fin_UTAS.shape[0]

    ## keep only first visit result if duplicate samples reported

    df_nonan = df_nonan.drop_duplicates(['PIN_Patient'], keep = 'first')
 
    #df_fin_UTAS.drop(['PIN_Patient','race','race_1','education','education_5'], inplace = True, axis = 1)

    #df_nonan.drop(['PIN_Patient'], inplace = True, axis = 1)

    return df_nonan

def predict_dilution(df_merged, cohort):
    'calculate predicted dilutions per cohort (UNM - Predcreatinine, DAR/NEU - PredSG)'

    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)

    df_merged = df_merged[df_merged['CohortType'] == cohort]

    #Where covariates = race + education + age + pre-pregnancy BMI + gestational age at sample collection + year of delivery + study site
    #these have to be in the dataset
    dilution_covars = ['race', 'education','babySex','BMI', 'ga_collection','birth_year']
    
    x_feature = 'age'
    #1) calculated specific gravity (or creatinine) Z-scores, <br>
    
    if cohort == 'NEU':
        orig_dilution = 'SPECIFICGRAVITY_V2'
        mean = df_merged['SPECIFICGRAVITY_V2'].mean()
        std =  df_merged['SPECIFICGRAVITY_V2'].std()
        
        df_merged['SPECIFICGRAVITY_V2_zscore'] = (df_merged['SPECIFICGRAVITY_V2'] -  mean) / std                                                     
        y_feature = 'SPECIFICGRAVITY_V2_zscore'
    
    if cohort == 'DAR':
        orig_dilution = 'urine_specific_gravity'
        mean = df_merged['urine_specific_gravity'].mean()
        std =  df_merged['urine_specific_gravity'].std()
        
        df_merged['urine_specific_gravity_zscore'] = (df_merged['urine_specific_gravity'] -  mean) / std                                                        
        y_feature = 'urine_specific_gravity_zscore'
        
    if cohort == 'UNM':
        orig_dilution = 'Creat_Corr_Result'
        mean = df_merged['Creat_Corr_Result'].mean()
        std =  df_merged['Creat_Corr_Result'].std()
        
        df_merged['Creat_Corr_Result_zscore'] = (df_merged['Creat_Corr_Result'] -  mean) / std                                                        
        y_feature = 'Creat_Corr_Result_zscore'
         
    data = add_confound(df_merged, x_feature, y_feature, dilution_covars)
    
    data.drop(['CohortType'], inplace = True, axis = 1)

    data['intercept'] = 1

    ids = data['PIN_Patient'].values

    data = data.select_dtypes(include = ['float','integer'])
     
    #data.drop(['PIN_Patient'], inplace = True, axis = 1)
    
    X = data[[x for x in data.columns if x !=y_feature and x!= 'PIN_Patient']]
    Y = data[y_feature]

  
    if X.shape[0] > 2:

        reg = sm.OLS(Y, X).fit() 
        ret = reg.summary()
    else:
        ret = 'error'

    fit_string = y_feature + '~'
    
    for x in X.columns:
        fit_string += ' + ' + str(x)
    
    fit_string = fit_string.replace('~ +','~')
    
    header = '<div> <b> Number samples: </b> ' + str(X.shape[0]) + '</div>'
    header += '<div> <b>  Model: </b>' + fit_string + '</div>'
    header += '<div> ===============================================</div>'

    Y_pred = reg.predict(X)
    
    #Y = original
    #Y_pred = prediction
    #X = dataset
    #ids = patient id
    #ret = summary fromm model..
    
    #pack predicted values with original and ids
    df_out = pd.DataFrame(list(zip(ids, Y, Y_pred)), columns = ['PIN_Patient','original', 'prediction'])

    data['PIN_Patient'] = ids
    
    bb = data.merge(df_out, on = 'PIN_Patient')
    
    # sanity check: make sure the index didn't get shifted
    assert bb[y_feature].values.tolist() == bb['original'].values.tolist()
    
    #4) back-transformed the Z-scores into the original units, <br>
    df_out['prediction_xvalue'] = df_out['prediction'] * std + mean
    df_out['original_xvalue'] = df_out['original'] * std + mean
    
    df_out2 = df_out.merge(df_merged[['PIN_Patient',orig_dilution]], on = 'PIN_Patient')
    df_out2 = df_out2.drop_duplicates(['PIN_Patient'])
    #5) calculated dilution ratios for each person by taking their predicted specific gravity (or creatinine) value and dividing it by their observed value, <br>
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



def runcustomanalysis():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ## Model 1: Restricted to participants with no fish/seafood consumption.

    ## Get data with no fish
    df_NEU = adapters.neu.get_dataframe_nofish()
    df_UNM = adapters.unm.get_dataframe_nofish()
    df_DAR = adapters.dar.get_dataframe_nofish()

    ## merge data frames
    df_NEUUNM = merge2CohortFrames(df_NEU,df_UNM)
    df_NEUDAR = merge2CohortFrames(df_NEU,df_DAR)
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)
    df_merged_3 = merge3CohortFrames(df_NEU,df_UNM,df_DAR)

    frames_for_analysis = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('NEUUNM', df_NEUUNM),
        ('NEUDAR', df_NEUDAR),
        ('UNMDAR', df_UNMDAR),
        ('UNMDARNEU', df_merged_3),
    ]

    for name, df in frames_for_analysis:
        print('Data Stats')
        print(name)
        print(df.shape)

    #set analysis parameters
    
    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    # set output paths for results:

    output_path_model1_adj = '/usr/src/app/mediafiles/analysisresults/model1adj/'
    output_path_model1_noadj = '/usr/src/app/mediafiles/analysisresults/model1noadj/'

    try:
        os.mkdir(output_path_model1_adj)
        os.mkdir(output_path_model1_noadj)
    except:
        print('Exists')

    # start analysis

    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model1_adj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")
            try:
                out = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model1_adj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()
    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model1_noadj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")


            try:

                out = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape
    
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model1_noadj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()


    #Model 2: Restricted to participants with arsenic speciation data.
    
    ## Get data with fish
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe()

    ## merge data frames
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)

    frames_for_analysis = [
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('UNMDAR', df_UNMDAR)
    ]

    for name, df in frames_for_analysis:
        print('Data Stats')
        print(name)
        print(df.shape)

    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    output_path_model2_adj = '/usr/src/app/mediafiles/analysisresults/model2adj/'
    output_path_model2_noadj = '/usr/src/app/mediafiles/analysisresults/model2noadj/'

    #output_path = '../mediafiles/analysisresults/'

    try:
        os.mkdir(output_path_model2_adj)
        os.mkdir(output_path_model2_noadj)
    except:
        print('Exists')

    for name, frame in frames_for_analysis:

       
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model2_adj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")


            try:

                out = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape
    
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model2_adj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()
    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model2_noadj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")


            try:

                out = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape
    
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model2_noadj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()


    #Model 3: Restricted to arsenic speciation data with AsB ≤1 µg/L.

    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model3_adj = '/usr/src/app/mediafiles/analysisresults/model3adj/'
    output_path_model3_noadj = '/usr/src/app/mediafiles/analysisresults/model3noadj/'
    #output_path = '../mediafiles/analysisresults/'

    try:
        os.mkdir(output_path_model3_adj)
        os.mkdir(output_path_model3_noadj)
    except:
        print('Exists')

    # remove the AsB <= 1
    df_UNM = df_UNM[df_UNM['UASB'] <= 1]
    df_DAR = df_DAR[df_DAR['UASB'] <= 1]
    df_UNMDAR_UASB = df_UNMDAR[df_UNMDAR['UASB'] <= 1]

    frames_for_analysis3 = [
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('UNMDAR', df_UNMDAR)
    ]

    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model3_adj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")
            try:
                out = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model3_adj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()

    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            text_file = open(output_path_model3_noadj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")
            try:
                out = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model3_noadj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()

   #Model 4: Restricted to arsenic speciation data with AsB ≤1 µg/L.

    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model4_adj = '/usr/src/app/mediafiles/analysisresults/model4adj/'
    output_path_model4_noadj = '/usr/src/app/mediafiles/analysisresults/model4noadj/'
    #output_path = '../mediafiles/analysisresults/'

    try:
        os.mkdir(output_path_model4_adj)
        os.mkdir(output_path_model4_noadj)
    except:
        print('Exists')

    ## Get data with fish
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe()

    frames_for_analysis3 = [
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('UNMDAR', df_UNMDAR)
    ]

    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            text_file = open(output_path_model4_adj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")
            try:
                out = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model4_adj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()

    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            text_file = open(output_path_model4_noadj + "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")
            try:
                out = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
                text_file.close()
            except Exception as e:
                text_file.write('Linear Regression Error:*\n')
                text_file.write(str(e))

            text_file.close()

        for y_feature in Y_features_binary:
            text_file = open(output_path_model4_noadj + "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), "w")

            try:
                out = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv')
                dims = frame.shape

                
                text_file.write(str(frame[all_vars + [y_feature]].describe()))
                text_file.write('\n')
                text_file.write("Number of participants: {}\n".format(dims[0]))
                text_file.write(str(out))
            
            except Exception as e:
                print('Logistic Regression Error:**')
                print(e)
                text_file.write('Logistic Regression Error:*\n')
                text_file.write(str(e))
            text_file.close()
   



    






