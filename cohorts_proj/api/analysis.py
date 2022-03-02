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
import sklearn

from datasets.models import RawFlower, RawUNM, RawDAR
from django.contrib.auth.models import User

#from api.dilutionproc import predict_dilution

from api import adapters

CAT_NHANES_ANALYTES = (
    ("UTAS", "Urinary Total Arsenic"),
    ("UALB_mg", "Urinary Albumin in mg/L"),
   # ("UALB_ug ", "Urinary Albumin in ug/L"),
    ("UCRT_mg", "Urinary Creatinine in mg/dL"),
    ("UCRT_umol", "Urinary Creatinine in umol/L"),
    ("UCR", "Urinary Chromium in ug/L"),
    #("I", "Urinary Iodine in ug/L"),
    ("UHG", "Urinary Mercury in ug/L"),
    ("UBA", "Urinary Barium in ug/L"),
    ("UCD", "Urinary Cadmium in ug/L"),
    ("UCO", "Urinary Cobalt in ug/L"),
    ("UCS", "Urianry Cesium in ug/L"),
    ("UMO","Urinary Molybdenum in ug/L"),
    ("UMN","Urinary Manganese in ug/L"),
    ("UPB","Urinary Lead in ug/L"),
    ("USB", "Urianry Antimony in ug/L"),
    ("USN","Urinary Tin in ug/L"),
    ("UTL", "Urinary Thallium in ug/L"),
    ("UTU", "Urinary Tungsten in ug/L"),
    ("UNI", "Urinary Nickel in ug/L"))

CAT_NEU_ANALYTES = [('Analytes', (
    ('USB', 'Antimony - Urine'),
    ('UTAS', 'Arsenic Total - Urine'), #modified just for poster - change back later/check if it's actually total 
    ('UBA', 'Barium - Urine'),
    ('UBE', 'Beryllium - Urine'),
    ('UCD', 'Cadmium - Urine'),
    ('UCS', 'Cesium - Urine'),
    ('UCR', 'Chromium - Urine'),
    ('UCO', 'Cobalt - Urine'),
    ('UCU', 'Copper - Urine'),
    ('UPB', 'Lead - Urine'),
    ('UMN', 'Manganese - Urine'),
    ('UHG', 'Mercury - Urine'),
    ('UMO', 'Molybdenum - Urine'),
    ('UNI', 'Nickel - Urine'),
    ('UPT', 'Platinum - Urine'),
    ('USE', 'Selenium - Urine'),
    ('UTL', 'Thallium - Urine'),
    ('USN', 'Tin - Urine'),
    ('UTU', 'Tungsten - Urine'),
    ('UUR', 'Uranium - Urine'),
    ('UVA', 'Vanadium - Urine'),
    ('UZN', 'Zinc - Urine')
    # Blood
    # ('BSB', 'Antimony - Blood'   ),
    # ('BTAS','Arsenic - Blood'    ),
    # ('BAL', 'Aluminum - Blood'   ),
    # ('BBE', 'Beryllium - Blood'  ),
    # ('BBA', 'Barium - Blood'     ),
    # ('BCD', 'Cadmium - Blood'    ),
    # ('BCS', 'Cesium - Blood'     ),
    # ('BCO', 'Cobalt - Blood'     ),
    # ('BCU', 'Copper - Blood'     ),
    # ('BCR', 'Chromium - Blood'   ),
    # ('BFE', 'Iron - Blood'       ),
    # ('BPB', 'Lead - Blood'       ),
    # ('BPB208','Lead (208) - Blood'),
    # ('BMB', 'Manganese - Blood'  ),
    # ('BHG', 'Mercury - Blood'    ),
    # ('BMO', 'Molybdenum - Blood' ),
    # ('BNI', 'Nickel - Blood'     ),
    # ('BPT', 'Platinum - Blood'   ),
    # ('BTL', 'Thallium - Blood'   ),
    # ('BTU', 'Tungsten - Blood'   ),
    # ('BUR', 'Uranium - Blood'    ),
    # ('BVA', 'Vanadium - Blood'   ),
    # ('BSE', 'Selenium - Blood'),
    # ('BSEG1124', 'Selenium+G1124 - Blood'),
    # ('BSN', 'Tin - Blood'        ),
    # ('BZN', 'Zinc - Blood'       ),
))]


CAT_DAR_ANALYTES = [('Analytes', (
    # Analyate acronym and name,                    Mapping in the dar DB
    ('UAG', ' Silver - Urine'),                     # Ag in ug/L
    ('UAL', ' Aluminium - Urine'),                  # Al in ug/L
    ('UCR',  'Chromium - Urine'),                   # Cr in ug/L
    ('UCU',  'Copper - Urine'),                     # Cu in ug/L
    ('UFE',  'Iron - Urine'),                       # Fe in ug/L
    ('UNI',  'Niquel - Urine'),                     # Ni in ug/L
    ('UVA',  'Vanadium - Urine'),                   # V in ug/L
    ('UZN',  'Zinc - Urine'),                       # Zn in ug/L
    # ('BCD',  'Cadmium - Blood'),
    # ('BHGE', 'Ethyl Mercury - Blood'),
    # ('BHGM', 'Methyl Mercury - Blood'),
    # ('BMN',  'Manganese - Blood'),
    # ('BPB',  'Lead - Blood'),
    # ('BSE',  'Selenium - Blood'),
    # ('IHG',  'Inorganic Mercury - Blood'),
    # ('THG',  'Mercury Total - Blood'),
    # ('SCU',  'Copper - Serum'),
    # ('SSE',  'Selenium - Serum'),
    # ('SZN',  'Zinc - Serum'),
    ('UAS3', 'Arsenous (III) acid - Urine'),        # As in ug/L
    # ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),              # AsB in ug/L
    # ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),                     # Ba in ug/L
    ('UBE',  'Beryllium - Urine'),                  # Be in ug/L
    ('UCD',  'Cadmium - Urine'),                    # Cd in ug/L
    ('UCO',  'Cobalt - Urine'),                     # Co in ug/L
    ('UCS',  'Cesium - Urine'),                     # Cs in ug/L
    ('UDMA', 'Dimethylarsinic Acid - Urine'),       # DMA in ug/L
    ('UHG',  'Mercury - Urine'),                    # Hg in ug/L
    # ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),     # MMA in ug/L
    ('UMN',  'Manganese - Urine'),                  # Mn in ug/L
    ('UMO',  'Molybdenum - Urine'),                 # Mo in ug/L
    ('UPB',  'Lead - Urine'),                       # PB in ug/L
    # ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),                   # Sb in ug/L
    ('USN',  'Tin - Urine'),                        # Sn in ug/L
    ('USR',  'Strontium - Urine'),                  # Sr in ug/L
    ('UTAS', 'Arsenic Total - Urine'),              # iAs in ug/L
    ('UTL',  'Thallium - Urine'),                   # Tl in ug/L
    # ('UTMO', 'Trimethylarsine - Urine')
    ('UTU',  'Tungsten - Urine'),                   # W in ug/L
    ('UUR',  'Uranium - Urine'),                    # U in ug/L

))]

CAT_UNM_ANALYTES = [('Analytes', (
    ('BCD',  'Cadmium - Blood'),
    ('BHGE', 'Ethyl Mercury - Blood'),
    ('BHGM', 'Methyl Mercury - Blood'),
    ('BMN',  'Manganese - Blood'),
    ('BPB',  'Lead - Blood'),
    ('BSE',  'Selenium - Blood'),
    ('IHG',  'Inorganic Mercury - Blood'),
    ('THG',  'Mercury Total - Blood'),
    ('SCU',  'Copper - Serum'),
    ('SSE',  'Selenium - Serum'),
    ('SZN',  'Zinc - Serum'),
    ('UAS3', 'Arsenous (III) acid - Urine'),
    ('UAS5', 'Arsenic (V) acid - Urine'),
    ('UASB', 'Arsenobetaine - Urine'),
    ('UASC', 'Arsenocholine - Urine'),
    ('UBA',  'Barium - Urine'),
    ('UBE',  'Beryllium - Urine'),
    ('UCD',  'Cadmium - Urine'),
    ('UCO',  'Cobalt - Urine'),
    ('UCS',  'Cesium - Urine'),
    ('UDMA', 'Dimethylarsinic Acid - Urine'),
    ('UHG',  'Mercury - Urine'),
    ('UIO',  'Iodine - Urine'),
    ('UMMA', 'Monomethylarsinic Acid - Urine'),
    ('UMN',  'Manganese - Urine'),
    ('UMO',  'Molybdenum - Urine'),
    ('UPB',  'Lead - Urine'),
    ('UPT',  'Platinum - Urine'),
    ('USB',  'Antimony - Urine'),
    ('USN',  'Tin - Urine'),
    ('USR',  'Strontium - Urine'),
    ('UTAS', 'Arsenic Total - Urine'),
    ('UTL',  'Thallium - Urine'),
    ('UTMO', 'Trimethylarsine - Urine'),
    ('UTU',  'Tungsten -  Urine'),
    ('UUR',  'Uranium - Urine'),


))]



def getOverviewPlot(df_neu, df_unm, df_dar, df_nhanes):
    
    unm = [x[0] for x in CAT_UNM_ANALYTES[0][1]]
    neu = [x[0] for x in CAT_NEU_ANALYTES[0][1]]
    dar = [x[0] for x in CAT_NEU_ANALYTES[0][1]]
    nhanes = [x[0] for x in CAT_NHANES_ANALYTES]
    print(nhanes)
    print(unm)

    for analyte in unm:
        if analyte not in df_unm.columns:
            df_unm[analyte] = 0

    main_idx = set(unm + neu + dar + nhanes)

    unm_har = []
    neu_har = []
    dar_har = []
    nhanes_har = []

    for x in main_idx:
        if x in unm: 
            unm_har.append(1)
        elif x not in unm:
            unm_har.append(0)
        if x in neu: 
            neu_har.append(1)
        elif x not in neu:
            neu_har.append(0)
        if x in dar: 
            dar_har.append(1)
        elif x not in dar:
            dar_har.append(0)
        if x in nhanes: 
            nhanes_har.append(1)
        elif x not in nhanes:
            nhanes_har.append(0)
            
    df = pd.DataFrame({'UNM':unm_har, 'NEU':neu_har, 'DAR':dar_har ,'NHANES':nhanes_har}, index = main_idx)
    df['sum'] = df.sum(axis = 1)

    df_2 = df.sort_values(by = 'sum')[['UNM','NEU','DAR','NHANES']]

    #melted = pd.melt(df2, id_vars=['index'], value_vars=['unm','neu','dar'])
    dff = pd.concat([df_neu,df_unm,df_dar,df_nhanes])

    dff = dff[['CohortType']+ list(main_idx)]
    dff_counts = dff.groupby(['CohortType']).count().transpose()
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(8, 13))

    df_counts = dff_counts.transpose()
    #masks = df2.transpose().drop(['CohortType'], axis = 0)

    res = sns.heatmap(df_counts.transpose(), 
                annot=True, 
                cbar =True,
                fmt="d", 
                linewidths=1,
                annot_kws={'rotation':0},
                ax=ax,cmap="PuBuGn")

    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 12)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=15, 
                    labelbottom = False, bottom=False, top = False, labeltop=True)

    plt.tight_layout()


    return res.figure

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

##Male/female infants
##Birth weight (g)
##Gestational age (weeks)
##SGA
##Preterm
##Birth length
##Smoke during pregnancy
##Specific gravity
##Creatinine
##Maternal BMI
##Parity
##Maternal level of education
## Total urinary arsenic (µg/L)
## Summation iAs + MMA + DMA (µg/L)
## Inorganic arsenic (µg/L)
## Monomethylarsonic acid (µg/L)
## Dimethylarsinic acid (µg/L)

def getCountsReport(df_neu,df_unm,df_dar):

    dff = pd.concat([df_neu,df_unm,df_dar])
    dff2 = dff.groupby(['CohortType']).count().transpose().reset_index()
    dff2['Total'] = dff2['NEU'] + dff2['DAR'] + dff2['UNM']

    return dff2


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
    
    #df_all = df_all.drop_duplicates(['CohortType','PIN_Patient','TimePeriod'])

    df_all = df_all.select_dtypes(include=['float64'])

    categorical = ['CohortType','TimePeriod','Outcome','folic_acid_supp', 'PIN_Patient',
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

    cohort = df['CohortType'].unique()
    categorical1 = ['CohortType','TimePeriod','Outcome','folic_acid_supp', 'PIN_Patient',
                'ethnicity','race','smoking','preg_complications','babySex','LGA','SGA','education']

    for var in categorical1:
        try:
            df[var] = df[var].astype(float)
        except:
            pass
   
    df22 = df[categorical1].drop_duplicates(['PIN_Patient'])
    categorical1.remove('PIN_Patient')
    df22 = df22[categorical1]
    melted = pd.melt(df22,id_vars=['CohortType'])    
    df33 = melted.groupby(['variable','value'])['value'].count()
    df33.index.names = ['variable', 'cat']
    
    return df33.reset_index()
    
def turntofloat(df):

    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    return df

def crude_reg(df_merged, x_feature, y_feature, covars, adjust_dilution, output, encode_cats):
    # inro for crude simple regression y = ax + b and report full results
    # y_feature has to be binary (i.e. 0,1)
    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    df_merged = df_merged[(~df_merged[x_feature].isna()) & (~df_merged[y_feature].isna())]
    #make sure all concentrations are above 0 - assuption is ok because lowest conc should have LOD
    df_merged = df_merged[(df_merged[x_feature]> 0) & (~df_merged[x_feature].isna())  ]
    split_covars = covars.split('|')
    print(split_covars)
    print(len(split_covars))
    ## adjust dilution
    if adjust_dilution == 'True':
        df_merged[x_feature] = df_merged[x_feature] / df_merged['UDR']

    if len(split_covars) > 1 & encode_cats == True:
        data = add_confound(df_merged, x_feature, y_feature, split_covars)
    if len(split_covars) > 1 & encode_cats == False:
        data = df_merged[[x_feature]+ [y_feature] + split_covars + ['CohortType']]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)

    else:
        data = df_merged[[x_feature]+ [y_feature] + ['CohortType']]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)
    ## problem - if we are using z_score to predict might be an issue
    
    data = data[(data[x_feature]> 0) & (~data[x_feature].isna())  ]
    
    data_copy = data.copy()
    data.drop(['CohortType'], inplace = True, axis = 1)

    data['intercept'] = 1
    #TODO: clean up - sometimes these variables are in the data, sometimes not (depends on selection )
    try:
        data['babySex'] = data['babySex'].astype(float)
    except:
        pass
    try:
        data['parity'] = data['parity'].astype(float)
    except:
        pass
  
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
    # info to display on webpage
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

def crude_logreg(df_merged, x_feature, y_feature, covars, adjust_dilution, output, encode_cats):
    # inro for crude simple logistic regression log(p(x)/1-p(x)) = ax + b and report slope, intercept, rvalue, plvalue, 'stderr
    # y_feature has to be binary (i.e. 0,1)
    df_merged = df_merged.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    df_merged = df_merged[(~df_merged[x_feature].isna()) & (~df_merged[y_feature].isna()) & \
        (df_merged[y_feature].isin([0.0,1.0,0,1, '0', '1', '0.0', '1.0']))]
    #make sure all concentrations are above 0 - assuption is ok because lowest conc should have LOD
    #df_merged = df_merged[df_merged[x_feature]> 0]

    #split the variables in the checkboxes
    split_covars = covars.split('|')
    print(split_covars)
    print(len(split_covars))
    ##adjust dilution
    if adjust_dilution == 'True':
        df_merged[x_feature] = df_merged[x_feature] / df_merged['UDR']

    if len(split_covars) > 1 & encode_cats == True:
        data = add_confound(df_merged, x_feature, y_feature, split_covars)
    if len(split_covars) > 1 & encode_cats == False:
        data = df_merged[[x_feature]+ [y_feature] + split_covars + ['CohortType']]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)
    else:
        data = df_merged[[x_feature]+ [y_feature] + ['CohortType']]
        data = data.dropna(axis = 'rows')
        data = turntofloat(data)

    data = data[(data[x_feature]> 0) & (~data[x_feature].isna())  ]
    data = data.dropna(how = 'any')
    data.drop(['CohortType'], inplace = True, axis = 1)
    # set intercept to 1
    data['intercept'] = 1

    #TODO: clean up
    try:
        data['babySex'] = data['babySex'].astype(float)
    except:
        pass
    try:
        data['parity'] = data['parity'].astype(float)
    except:
        pass

    data = data.select_dtypes(include = ['float','integer'])
    print('Data shape after intselect')

    #independent
    X = data[[x for x in data.columns if x !=y_feature and x!= 'PIN_Patient']]
    #target
    Y = data[y_feature]
    #log of the exposure
    X[x_feature]= np.log(X[x_feature])

    # fit the model
    if df_merged.shape[0] > 1:
        log_reg = sm.Logit(Y, X).fit()
        ret = log_reg.summary()
    else:
        ret = 'error'
    # fit string for site
    fit_string = y_feature + '~'
    
    for x in X.columns:
        if x == x_feature:
            fit_string += ' + log(' + str(x) +')'
        else:

            fit_string += ' + ' + str(x)
    
    fit_string = fit_string.replace('~ +',' ~')
    header = ' <div><b> Logistic Regression </b> </div>'
    header += '<div><b> Number samples: </b> ' + str(X.shape[0]) + '</div>'
    header += '<div><b> Model: </b>' + fit_string + '</div>'
    header += '<div><b> Group: </b> CohortType '
    
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

def verifyclean(df):
    df = df.replace(-9,np.nan).replace('-9',np.nan).replace(999,np.nan).replace(888,np.nan)
    return df


def add_confound(df_merged, x_feature, y_feature, conf):
    print(df_merged.shape)
    # check if confounders are added
    if len(conf) > 1:
        cols_to_mix =  [x_feature, y_feature, 'PIN_Patient', 'CohortType'] + conf
    else:
        cols_to_mix = [x_feature, y_feature, 'PIN_Patient', 'CohortType']

    # drop any missing values as mixed model requires complete data
    df_nonan = df_merged[cols_to_mix].dropna(axis='rows')
    #df_nonan['smoking'] = df_nonan['smoking'].astype(int)
    print(df_nonan.shape)

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

    if 'race' in conf: df_nonan = add_cats('race', df_nonan, '1')
    if 'smoking' in conf: df_nonan = add_cats('smoking', df_nonan, '0')
    if 'education' in conf: df_nonan = add_cats('education', df_nonan, '5')


    return df_nonan

##text file writing function to shorten length of the code
def text_writing(name, frame, x_feat, y_feat, all_variables, path, output, txt_file_specifics, reg_type):
    try:
        text_file = open(path + txt_file_specifics, "w")
        dims = frame.shape
        text_file.write(str(frame[all_variables + [y_feat]].describe()))
        text_file.write('\n')
        text_file.write("Number of participants: {}\n".format(dims[0]))
        text_file.write(str(output))
        text_file.close()
    except Exception as e:
        text_file.write(reg_type + ' Error:*\n')
        text_file.write(str(e))
    text_file.close()

## main analysis
## with categories encoded
def runcustomanalysis1():

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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature), 'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')



    #Model 2: Restricted to participants with arsenic speciation data.
    
    ## Get data with fish
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

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
            output= crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')
    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')



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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


   #Model 4: Sensitivity analysis 

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

    ## Get data all
    df_NEU = adapters.neu.get_dataframe()
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

    ## merge data frames
    df_NEUUNM = merge2CohortFrames(df_NEU,df_UNM)
    df_NEUDAR = merge2CohortFrames(df_NEU,df_DAR)
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)
    df_merged_3 = merge3CohortFrames(df_NEU,df_UNM,df_DAR)

    frames_for_analysis4 = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('NEUUNM', df_NEUUNM),
        ('NEUDAR', df_NEUDAR),
        ('UNMDAR', df_UNMDAR),
        ('UNMDARNEU', df_merged_3),
    ]

    for name, frame in frames_for_analysis4:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "logistic_reg{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


# run crude models mo confounders
def runcustomanalysis2():

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
    covars = ''
    all_vars = [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    # set output paths for results:

    output_path_model1_adj = '/usr/src/app/mediafiles/analysisresults/crudemodel1adj/'
    output_path_model1_noadj = '/usr/src/app/mediafiles/analysisresults/crudemodel1noadj/'

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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')



    #Model 2: Restricted to participants with arsenic speciation data.
    
    ## Get data with fish
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

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
    covars = ''
    all_vars = [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    output_path_model2_adj = '/usr/src/app/mediafiles/analysisresults/crudemodel2adj/'
    output_path_model2_noadj = '/usr/src/app/mediafiles/analysisresults/crudemodel2noadj/'

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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')



    #Model 3: Restricted to arsenic speciation data with AsB ≤1 µg/L.

    x_feature = 'UTAS'
    covars = ''
    all_vars = [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model3_adj = '/usr/src/app/mediafiles/analysisresults/crudemodel3adj/'
    output_path_model3_noadj = '/usr/src/app/mediafiles/analysisresults/crudemodel3noadj/'
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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


   #Model 4: Sensitivity analysis 

    x_feature = 'UTAS'
    covars = ''
    all_vars = [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model4_adj = '/usr/src/app/mediafiles/analysisresults/crudemodel4adj/'
    output_path_model4_noadj = '/usr/src/app/mediafiles/analysisresults/crudemodel4noadj/'
    #output_path = '../mediafiles/analysisresults/'

    try:
        os.mkdir(output_path_model4_adj)
        os.mkdir(output_path_model4_noadj)
    except:
        print('Exists')

    ## Get data with no fish
    df_NEU = adapters.neu.get_dataframe()
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

    ## merge data frames
    df_NEUUNM = merge2CohortFrames(df_NEU,df_UNM)
    df_NEUDAR = merge2CohortFrames(df_NEU,df_DAR)
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)
    df_merged_3 = merge3CohortFrames(df_NEU,df_UNM,df_DAR)

    frames_for_analysis4 = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('NEUUNM', df_NEUUNM),
        ('NEUDAR', df_NEUDAR),
        ('UNMDAR', df_UNMDAR),
        ('UNMDARNEU', df_merged_3),
    ]

    for name, frame in frames_for_analysis4:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', True)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


def runcustomanalysis3():

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

    output_path_model1_adj = '/usr/src/app/mediafiles/analysisresults/model1adjnoenc/'
    output_path_model1_noadj = '/usr/src/app/mediafiles/analysisresults/model1noadjnoenc/'

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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model1_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')



    #Model 2: Restricted to participants with arsenic speciation data.
    
    ## Get data with fish
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

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
    
    output_path_model2_adj = '/usr/src/app/mediafiles/analysisresults/model2adjnoenc/'
    output_path_model2_noadj = '/usr/src/app/mediafiles/analysisresults/model2noadjnoenc/'

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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    
    #without adjustment
    
    for name, frame in frames_for_analysis:

        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model2_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    #Model 3: Restricted to arsenic speciation data with AsB ≤1 µg/L.

    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model3_adj = '/usr/src/app/mediafiles/analysisresults/model3adjnoenc/'
    output_path_model3_noadj = '/usr/src/app/mediafiles/analysisresults/model3noadjnoenc/'
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
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')


    #no adj
    for name, frame in frames_for_analysis3:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        
        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model3_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

   #Model 4: Sensitivity analysis 

    x_feature = 'UTAS'
    covars = 'babySex|BMI|parity|smoking|education'
    all_vars = covars.split('|') + [x_feature] 
    Y_features_continous = ['Outcome_weeks','birthWt', 'headCirc', 'birthLen']
    Y_features_binary    = ['LGA','SGA','Outcome']
    
    ## Number of Participants
    output_path_model4_adj = '/usr/src/app/mediafiles/analysisresults/model4adjnoenc/'
    output_path_model4_noadj = '/usr/src/app/mediafiles/analysisresults/model4noadjnoenc/'
    #output_path = '../mediafiles/analysisresults/'

    try:
        os.mkdir(output_path_model4_adj)
        os.mkdir(output_path_model4_noadj)
    except:
        print('Exists')

    ## Get data with fish
    df_NEU = adapters.neu.get_dataframe()
    df_UNM = adapters.unm.get_dataframe()
    df_DAR = adapters.dar.get_dataframe_pred()

    ## merge data frames
    df_NEUUNM = merge2CohortFrames(df_NEU,df_UNM)
    df_NEUDAR = merge2CohortFrames(df_NEU,df_DAR)
    df_UNMDAR = merge2CohortFrames(df_UNM,df_DAR)
    df_merged_3 = merge3CohortFrames(df_NEU,df_UNM,df_DAR)

    frames_for_analysis4 = [
        ('NEU', df_NEU),
        ('UNM', df_UNM),
        ('DAR', df_DAR),
        ('NEUUNM', df_NEUUNM),
        ('NEUDAR', df_NEUDAR),
        ('UNMDAR', df_UNMDAR),
        ('UNMDARNEU', df_merged_3),
    ]

    for name, frame in frames_for_analysis4:
        
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]
        print('Min: {} Max: {}'.format(frame['UTAS'].min(), frame['UTAS'].max()))

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')

        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'True', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_adj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')

    #no adj
    for name, frame in frames_for_analysis3:
        
        frame = frame[(frame['UTAS'] > 0) & (~frame['UTAS'].isna())]

        for y_feature in Y_features_continous:
            output = crude_reg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "linear_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Linear Regression')


        for y_feature in Y_features_binary:
            output = crude_logreg(frame, x_feature, y_feature, covars, 'False', 'csv', False)
            text_writing(name, frame, x_feature, y_feature, all_vars, output_path_model4_noadj, output, "logistic_reg_{}_{}_log({}).txt".format(name, y_feature, x_feature),'Logistic Regression')
