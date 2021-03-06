import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from api import analysis
import matplotlib
import traceback
# Functions to generate different types of plots

# ==============================================================================
# Common functions

def getInfoString(data, x_feature, y_feature, color_by):
    """Return high level statistics of unique samples."""

    filtered_df = data[data[[x_feature, y_feature]].notnull().all(1)]
    info1 = str(filtered_df[[x_feature, y_feature]].describe(include='all'))
    info2 = str(filtered_df[[color_by]].describe(include='all'))

    info = "Summary of intersection between analytes\n" + \
        "(Not Null samples only):\n\n" + \
        info1+"\n\n"+info2

    return info



def getHistInfoString(data, feature):
    """Return high level statistics of unique samples for histogram plot."""

    filtered_df = data[data[[feature]].notnull().all(1)]
    info1 = str(filtered_df[[feature]].describe(include='all'))
    info = "Summary of intersection between analytes\n" + \
        "(Not Null samples only):\n\n" + \
        info1+"\n\n"

    return info

def getKdeInfoString(data, feature,color_by):
    """Return high level statistics of unique samples for kde plot."""

    filtered_df = data[data[[feature]].notnull().all(1)]
    cohorts = data['CohortType'].unique().tolist()
    
    df = filtered_df.groupby(['CohortType']).agg({feature:
        ['count',np.mean,np.var]}).reset_index()

    info1 = str(df)

    info = "Summary \n" \
        "(Not Null samples only):\n\n" + \
        info1+"\n\n"

    return info
    
def getViolinCatInfoString(data, x_feature,y_feature, color_by):
    """Return high level statistics of unique samples for violin plot."""

    filtered_df = data[data[[x_feature]].notnull().all(1)]

    cohorts = data['CohortType'].unique().tolist()

    df = data.groupby([x_feature,color_by]).agg({y_feature:
        ['count',np.mean,np.var]}).reset_index()

    info1 = str(df)

    info = "Summary \n" \
        "(Not Null samples only):\n\n" + \
        info1+"\n\n"

    return info

def addInfoToAxis(info, ax, id=1):
    """Add info to axis ax, at position id."""
    sns.despine(ax=ax[id], left=True, bottom=True, trim=True)
    ax[id].set(xlabel=None)
    ax[id].set(xticklabels=[])

    ax[id].text(0, 0, info, style='italic',
                bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10},
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax[1].transAxes)

def noDataMessage():
    info = 'Error: There are no samples matching the criteria for\n' + \
        'the dataset, features, and filters selected.\n\n' + \
        'Solution: Select a different query combination.'

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    sns.despine(ax=ax, left=True, bottom=True, trim=True)
    ax.text(0.5, .5, info, style='italic', fontsize='large',
            bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10},
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

    return fig
# ==============================================================================
# Plots without statistics

def getScatterPlot(data, x_feature, y_feature, color_by):
    fig, _ = plt.subplots(1, 1, figsize=(5, 5))
    sns.scatterplot(
        data=data, x=x_feature, y=y_feature,
        hue=color_by, alpha=0.8, s=15, style='CohortType')

    return fig


def getPairPlot(data, x_feature, y_feature, color_by):
    gr = sns.pairplot(
        data, vars=[x_feature, y_feature], hue=color_by, height=3)

    return gr


def getCatPlot(data, x_feature, y_feature, color_by):

    gr = sns.catplot(data=data, x=x_feature,
                     y=y_feature, hue=color_by)

    return gr


def getViolinCatPlot(data, x_feature, y_feature, color_by):
    
    #data[y_feature] = np.log(data[y_feature])+
    ##get standard deviation
    ##filter 
    std = np.std(data[y_feature])
    data_rem = data.loc[data[y_feature] < 2*std]

    if data_rem.shape[0] > 20:
        data_c = data_rem
    else:
        data_c = data

    gr = sns.catplot(data=data_c, x=x_feature,
                     y=y_feature, hue=color_by, kind="violin")

    return gr

def getHistogramPlot(data, x_feature, y_feature, color_by):
    fig, _ = plt.subplots(1, 1, figsize=(5, 5))

    sns.distplot(data[x_feature])
    return fig

def getMLPlot(data, x_feature, y_feature, color_by):

    mixed_ml_info = analysis.crude_mixedML(data, x_feature, y_feature)
    sns.set(style = 'white')

    plt.figure(figsize = (5,5))

    fig, ax = plt.subplots(1, 1, figsize=(5*2, 5))

    ax.text(0, 0, mixed_ml_info, style='italic',
                bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10},
                horizontalalignment='left',
                verticalalignment='bottom')
    
    sns.despine(left = False, bottom = False)

    return fig


def getbinomialMLPlot(data, x_feature, y_feature, color_by):

    mixed_ml_info = analysis.crude_binomial_mixedML(data, x_feature, y_feature)

    sns.set(style = 'white')

    plt.figure(figsize = (5,5))

    fig, ax = plt.subplots(1, 1, figsize=(5*2, 5))

    ax.text(0, 0, mixed_ml_info, style='italic',
                bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10},
                horizontalalignment='left',
                verticalalignment='bottom')
    
    sns.despine(left = False, bottom = False)

    return fig

def getlogistcRegPlot(data, x_feature, y_feature, color_by):

    mixed_ml_info = analysis.crude_logreg(data, x_feature, y_feature)

    print(mixed_ml_info)

    fig, ax = plt.subplots(1, 2, figsize=(5*2, 5))

    sns.set()

    #plt.figure(figsize = (5,5))

    #fig, ax = plt.subplots(1, 2, figsize=(5*2, 5))

    data['log_' + x_feature] = np.log(data[x_feature] )
    
    gr = sns.regplot(data=data, x='log_' + x_feature,
                     y=y_feature, logistic = True,  ax=ax[0])
                     

    addInfoToAxis(mixed_ml_info, ax)

    #ax.text(0, 0, mixed_ml_info, style='italic',
    #            bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10},
    #            horizontalalignment='left',
     #           verticalalignment='bottom')
    
    sns.despine(left = False, bottom = False)

    return fig

def getRegPlot(data, x_feature, y_feature, color_by):

    sns.set()

    plt.figure(figsize = (5,5))

    #fig, ax = plt.subplots(1, 2, figsize=(5*2, 5))

    data['log_' + x_feature] = np.log(data[x_feature] )
    
    gr = sns.regplot(data=data, x='log_' + x_feature,
                     y=y_feature)

    x = data['log_' + x_feature].values
    y = data[y_feature].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=x,
        y=y)
    
    reg_info = "y={:.2f}x + {:.2f} \n r^2={:.2f}, p={:.2f}, std_err={:.2f}\n".format(
    slope, intercept, r_value**2, p_value, std_err)
    #plt.suptitle(''.join(reg_info), **{'x': 1.4, 'y':.98})
    gr.set_title(reg_info,**{'x': .5, 'y':.98})

    return gr.figure




def getRegColorPlot(data, x_feature, y_feature, color_by):

    data = fixvisits(data)

    sns.set()

    filtered_df = data[data[[x_feature, y_feature]].notnull().all(1)]

    #take log transform
    filtered_df[x_feature] = np.log(filtered_df[x_feature])
    data[x_feature] = np.log(data[x_feature])
  
    color_by_options = filtered_df[color_by].unique()

    reg_info0 = ''
    reg_info1 = ''

    gr = sns.lmplot(data=filtered_df, x=x_feature,
                    y=y_feature, hue=color_by, legend_out=False)
    
    #TODO fix legends - not the best way to display equations

    labs = list(gr._legend_data.keys())
    labs2 = []

    for x in labs:

        try:
            x2 = float(x)
        except:
            x2 = x
        labs2.append(x2)

    print(labs)
    # for x in xrange(len(color_by_options)):
    # adding support if we want to color by multiple options 

    num_lines = len(gr.axes.flat[0].get_lines())

    reg_infos = ['Regression Equations:\n']

    if (num_lines > 1):

        for i in range(0, num_lines):
            try:
                
                x = filtered_df.loc[filtered_df[color_by] == labs2[i], x_feature].values
                y = filtered_df.loc[filtered_df[color_by] == labs2[i], y_feature].values
               
                # this is problematic for calcuatign the true stats
                # = [gr.axes.flat[0].get_lines()[i].get_xdata()]
                #y = [gr.axes.flat[0].get_lines()[i].get_ydata()]

                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
              
                reg_info1 = str(labs[i]) + ": y={:.2f}x + {:.2f} \n r^2={:.2f}, p={:.2f}, std_err={:.2f}\n".format(
                slope, intercept, r_value**2, p_value, std_err)

                reg_infos.append(reg_info1)

            except Exception as exc:
                print('Error: We need 2 points to create a line...')
                print(traceback.format_exc())
                print(exc)

    #reg_info = "{}  |  {}".format(reg_info0, reg_info1)

    #gr.fig.suptitle(' | '.join(reg_infos), 1, .98)

    plt.suptitle(''.join(reg_infos), **{'x': 1.4, 'y':.98})
    plt.legend(bbox_to_anchor=(.85, 0., 0.5, 0.5), title = color_by)

    return gr


def getRegDetailedPlot(data, x_feature, y_feature, color_by):


    def get_stats(x, y):
        """Prints more statistics"""

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=x, y=y)
        reg_info = "f(x)={:.2f}x + {:.2f} \nr^2={:.2f} p={:.2f}".format(
            slope, intercept, r_value, p_value)

        # TODO return value is incompatible with jointplot stat_func
        return reg_info

    def r_squared(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    gr = sns.jointplot(data=data, x=x_feature,
                       y=y_feature, kind="reg", stat_func=r_squared)

    return gr

# ==============================================================================
# Plots with statistics


def getIndividualScatterPlotWithInfo(data, x_feature, y_feature, color_by):

    info = getInfoString(data, x_feature, y_feature, color_by)

    color_by_options = data[color_by].unique()
    color_by_count = len(color_by_options)
    fig, ax = plt.subplots(1, color_by_count+1,
                           sharey=True, figsize=(5*(color_by_count+1), 5))

    for i, v in enumerate(color_by_options):
        if i > 0:
            sns.scatterplot(
                data=data[data[color_by] == v], x=x_feature, y=y_feature,
                hue=color_by, alpha=0.8, s=20, hue_order=color_by_options,
                legend=False, style='CohortType', ax=ax[i])
        else:  # With legend
            sns.scatterplot(
                data=data[data[color_by] == v], x=x_feature, y=y_feature,
                hue=color_by, alpha=0.8, s=20, hue_order=color_by_options,
                legend='brief', style='CohortType', ax=ax[i])
        ax[i].set_title(str(color_by)+': '+str(v))

    sns.despine(ax=ax[color_by_count], left=True, bottom=True, trim=True)
    ax[color_by_count].set(xlabel=None)
    ax[color_by_count].set(xticklabels=[])

    ax[color_by_count].text(0, 0, info, style='italic',
                            bbox={'facecolor': 'azure', 'alpha': 1.0, 'pad': 10})

    return fig


def getScatterPlotWithInfo(data, x_feature, y_feature, color_by):
    info = getInfoString(data, x_feature, y_feature, color_by)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5*2, 5))

    sns.scatterplot(
        data=data, x=x_feature, y=y_feature,
        hue=color_by, alpha=0.8, s=15, style='CohortType', ax=ax[0])

    addInfoToAxis(info, ax)

    return fig


def getHistogramPlotWithInfo(data, x_feature, y_feature, color_by):

    sns.set()

    info = getKdeInfoString(data, x_feature, color_by)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5*2, 5))

    std = np.std(data[x_feature])

    data_rem = data.loc[data[x_feature] < 2* std]

    if data_rem.shape[0] > 20:
        data_c = data_rem
    else:
        data_c = data

    sns.distplot(data_c[x_feature], ax=ax[0])

    addInfoToAxis(info, ax) 

    return fig

def getKdePlotWithInfo(data, x_feature, y_feature, color_by):

    data = fixvisits(data)
    
    sns.set()

    info = getKdeInfoString(data, x_feature, color_by)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5*2, 5))

    std = np.std(data[x_feature])

    data_rem = data.loc[data[x_feature] < 2* std]

    if data_rem.shape[0] > 20:
        data_c = data_rem
    else:
        data_c = data

    #sns.distplot(d_outliers, ax=ax[0])
    ##kdeplot temprary substitution for histogram
    b = sns.kdeplot(
        data=data_c, x=x_feature, hue=color_by,
        fill=True, common_norm=False, 
        alpha=.5, linewidth=0, ax = ax[0]
     )    

    ax[0].set(xlim=(0,None))

    addInfoToAxis(info, ax) 

    return fig

def getViolinCatPlotWithInfo(data, x_feature, y_feature, color_by):

    info = getViolinCatInfoString(data, x_feature, y_feature,color_by)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5*2, 5))

    std = np.std(data[y_feature])
    data_rem = data.loc[data[y_feature] < 2*std]

    if data_rem.shape[0] > 20:
        data_c = data_rem
    else:
        data_c = data
    
    
    sns.violinplot(data=data_c, x=x_feature,
                     y=y_feature, 
                     hue=color_by, 
                     scale = 'width',
                     kind="box", 
                     ax = ax[0],
                     linewidth = .58,
                     split = False)
    
    addInfoToAxis(info, ax) 

    return fig

def vertical_mean_line(x, **kwargs):
    ls = {"0":"-","1":"--"}
    plt.axvline(x.mean(), linestyle =ls[kwargs.get("label","0")], 
                color = kwargs.get("color", "g"))
    txkw = dict(size=12, color = kwargs.get("color", "g"), rotation=90)
    tx = "mean: {:.2f}, std: {:.2f}".format(x.mean(),x.std())
    plt.text(x.mean()+1, 0.052, tx, **txkw)

def getCustomFacetContinuousPlot1(df_merged, x_feature, y_feature, time_period, type):

    print(time_period)
    #if time_period != 9:

    #    df_merged = df_merged[df_merged['TimePeriod']==time_period]

    #['age','BMI','fish','birthWt','birthLen','WeightCentile',
    #'Outcome_weeks','ga_collection'] 

    if type == 0:

        continuous = ['age','BMI','fish','birthWt','birthLen','WeightCentile',
                    'Outcome_weeks','ga_collection'] 
    if type == 1:

        continuous = ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 

    df_merged_copy = df_merged.copy()

    for x in ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA']:
        
        df_merged_copy[x] = np.log(df_merged_copy[x])

    data = pd.melt(df_merged_copy[continuous + ['CohortType']],id_vars=['CohortType'], var_name = 'x')

    data.loc[data['value'].isin([97,888,999,-9]),'value'] = np.nan

    sns.set(font_scale = 1.5)

    g = sns.FacetGrid(data, col="x", 
                col_wrap=4, sharex = False, sharey = False, legend_out = False)

    bp = g.map_dataframe(sns.histplot, x="value",             
                        common_norm = True, 
                        hue = 'CohortType',
                        legend = False,
                        common_bins = True)

        # The color cycles are going to all the same, doesn't matter which axes we use
    Ax = bp.axes[0]

  

    for ax in bp.axes.ravel():
        ax.legend()

    #print(Ax.get_children())

    # Some how for a plot of 5 bars, there are 6 patches, what is the 6th one?
    Boxes = [item for item in Ax.get_children()
         if isinstance(item, matplotlib.patches.Rectangle)][:-1]

    
    Boxes = list(set(Boxes))

    colors = []

    for box in Boxes:
        colors.append(box.get_facecolor())
    
    colors = list(set(colors))


    legend_labels  = ['UNM','NEU','DAR'] 


    # Create the legend patches
    legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                  C, L in zip(colors,
                              legend_labels)]


    plt.legend(handles=legend_patches, loc = 1, bbox_to_anchor=(2, 1))

    #g.add_legend()
    
    g.set_xticklabels(rotation=90, size = 12)

    axes = g.axes.flatten()

    for ax in axes:

        ax.set_ylabel("Count")

    # The color cycles are going to all the same, doesn't matter which axes we use
    
 
    return g

def getCustomFacetCategoricalPlot1(df_merged, x_feature, y_feature, time_period):

    categorical = ['CohortType','TimePeriod','folic_acid_supp',
                'ethnicity','race','smoking','preg_complications',
                'babySex','Outcome','LGA','SGA']

    df_merged = df_merged[categorical +['PIN_Patient']].drop_duplicates(['PIN_Patient'])

    for x in categorical:
        try:
            df_merged[x] = df_merged[x].astype(str)
        except:
            print(x)

    conditions = [
        (df_merged['babySex'] == '1.0'),
        (df_merged['babySex'] == '2.0'),
        (df_merged['babySex'] == '3.0'),
        (df_merged['babySex'] == 'NaN')
    ]

    choices = ['M','F','A','Miss']

    df_merged['babySex'] = np.select(conditions, choices, default='-9')

    conditions = [
        (df_merged['race'] == '1.0'),
        (df_merged['race'] == '2.0'),
        (df_merged['race'] == '3.0'),
        (df_merged['race'] == '4.0'),
        (df_merged['race'] == '5.0'),
        (df_merged['race'] == '6.0'),
        (df_merged['race'] == '97'),
        (df_merged['race'] == '888'),
        (df_merged['race'] == '999')
    ]

      
    choices =  ['Whte', 'AfrA', 'AIAN', 'Asian','NHPI', 'Mult', 'Oth', 'Ref', 'DK']

    df_merged['race'] = np.select(conditions, choices, default='-9')


    conditions = [
        (df_merged['ethnicity'] == '1.0'),
        (df_merged['ethnicity'] == '2.0'),
        (df_merged['ethnicity'] == '3.0'),
        (df_merged['ethnicity'] == '4.0'),
        (df_merged['ethnicity'] == '5.0'),
        (df_merged['ethnicity'] == '6.0'),
        (df_merged['ethnicity'] == '97'),
        (df_merged['ethnicity'] == '888'),
        (df_merged['ethnicity'] == '999')
    ]

      
    choices =  ['PR', 'Cuban', 'Domin.', 'Mex.','MexA', 'SouthA', 'Oth', 'Ref', 'DK']

    CAT_NEU_SMOKING = [
    ('0', 'never smoked'),
    ('1', 'past smoker'),
    ('2', 'current smoker'), 
    ('3', 'smoke during pregnancy')
    ]

    df_merged['ethnicity'] = np.select(conditions, choices, default='-9')


    conditions = [
        (df_merged['smoking'] == '0.0'),
        (df_merged['smoking'] == '1.0'),
        (df_merged['smoking'] == '2.0'),
        (df_merged['smoking'] == '3.0'),
       
    ]

    choices =  ['Never', 'past', 'curr', 'Pregsmk']

    df_merged['smoking'] = np.select(conditions, choices, default='Miss')


    conditions = [
        (df_merged['folic_acid_supp'] == '0.0'),
        (df_merged['folic_acid_supp'] == '1.0'),
        (df_merged['folic_acid_supp'] == '999.0'),
    ]
    choices =  ['No','Yes','Ref']

    df_merged['folic_acid_supp'] = np.select(conditions, choices, default='Miss')


    conditions = [
        (df_merged['preg_complications'] == '0.0'),
        (df_merged['preg_complications'] == '1.0'),
        (df_merged['preg_complications'] == '999.0'),
    ]
    choices =  ['No','Yes','Ref']

    df_merged['preg_complications'] = np.select(conditions, choices, default='Miss')

    conditions = [
        (df_merged['Outcome'] == '0.0'),
        (df_merged['Outcome'] == '1.0'),
        (df_merged['Outcome'] == '999.0'),
    ]
    choices =  ['FullTerm','Preterm','Miss']

    df_merged['Outcome'] = np.select(conditions, choices, default='Miss')


    conditions = [
        (df_merged['LGA'] == '0.0'),
        (df_merged['LGA'] == '1.0'),
        (df_merged['LGA'] == '999.0'),
    ]
    choices =  ['No','Yes','Miss']

    df_merged['LGA'] = np.select(conditions, choices, default='Miss')

    conditions = [
        (df_merged['SGA'] == '0.0'),
        (df_merged['SGA'] == '1.0'),
        (df_merged['SGA'] == '999.0'),
    ]
    choices =  ['No','Yes','Miss']

    df_merged['SGA'] = np.select(conditions, choices, default='Miss')

    data = pd.melt(df_merged[categorical],id_vars=['CohortType'], var_name = 'x')

    data.loc[data['value'].isin(['97','888','999','-9']),'value'] = 'Miss'

    sns.set(font_scale = 1)

    print(data.columns)

    g = sns.FacetGrid(data, col="x", 
                    col_wrap=4, sharex = False, sharey = False)

    bp = g.map_dataframe(sns.histplot, x="value", 
                        common_norm = False, 
                        common_bins = True, multiple = "dodge", hue = 'CohortType').add_legend()

    
    # The color cycles are going to all the same, doesn't matter which axes we use
    Ax = bp.axes[0]

    # Some how for a plot of 5 bars, there are 6 patches, what is the 6th one?
    Boxes = [item for item in Ax.get_children()
         if isinstance(item, matplotlib.patches.Rectangle)][:-1]

    legend_labels  = ['UNM','UNM','UNM','NEU', 'NEU','NEU','DAR','DAR','DAR']


    # Create the legend patches
    legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
                  C, L in zip([item.get_facecolor() for item in Boxes],
                              legend_labels)]

    legend_patches = legend_patches[0::3]
    
    plt.legend(handles=legend_patches, loc = 1, bbox_to_anchor=(2, 1))

    #g.add_legend()
    
    g.set_xticklabels(rotation=90, size = 12)

    axes = g.axes.flatten()

    for ax in axes:

        ax.set_ylabel("Count")

    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    
    return g

def fixvisits(data):

    data = data[(data['ga_collection'] > 13) & (data['ga_collection'] < 28)]
    data = data.drop_duplicates(['PIN_Patient'], keep = 'first')

    return data

def getCustomFacetLMPlot1(df_merged, x_feature, y_feature, time_period):

    df_merged = fixvisits(df_merged)


    categorical = ['CohortType','TimePeriod','folic_acid_supp',
                'ethnicity','race','smoking','preg_complications','babySex','Outcome','LGA','SGA']

    df_merged_copy = df_merged.copy()

    continuous =  ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 

    for x in ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA']:
        
        df_merged_copy[x] = np.log(df_merged_copy[x])
        
    data = pd.melt(df_merged_copy[continuous+['CohortType'] +[y_feature]],id_vars=['CohortType',y_feature], var_name = 'variable')

    data = data[data['variable']!='PIN_Patient']

    data.loc[data['value'].isin([97,888,999,-9]),'value'] = np.nan

    data = data[data[y_feature] > 0]
    
    sns.set(font_scale = 1.5,style = 'whitegrid')


    g = sns.lmplot(y=y_feature, 
                    x="value", hue="CohortType", 
                    col="variable", col_wrap = 7,
                   scatter_kws={"s": 25},
                data=data, x_jitter=.1, sharex = False, sharey = True)

    return g

def getCorrelationHeatmap(data):

    data = fixvisits(data)

    arsenic_cont = ['UTAS','UIAS','UASB', 'UAS3', 'UAS5', 'UDMA','UMMA'] 
    to_corr_cols = ['Outcome_weeks','age','BMI','fish','birthWt','birthLen','WeightCentile'] + arsenic_cont


    for col in arsenic_cont:

        if col not in data.columns:

            data[col] = np.nan

    
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
    p_values = analysis.corr_sig(data[to_corr_cols])
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
    return g.figure

def getClusterMap(data, color_by):

    data = fixvisits(data)
    data = data[~data[color_by].isna()]

    for col in data.columns:

        try:
            data[col] = data[col].astype(float)
            data.loc[data[x] < 0, x] = np.nan
        except:
            data[col] = data[col]

    #this clustermap is still in testing so will only fit it for specific analytes


    analytes = ['UTAS', 'UBA', 'USN', 'UPB', 'UBE', 'UUR', 'UTL', 'UHG', 'UMO',  'UMN', 'UCO']
    #analytes = ['UTAS'] + ['Outcome_weeks','age','BMI','fish','birthWt','birthLen','WeightCentile'] 
    print('before')
    print(data.shape)
    print(data[color_by].unique())

    X = data[[color_by] + analytes].dropna( how = 'any', axis = 'rows')

    print('after')
    print(X.shape)
    print(X[color_by].unique())

    labels = X[color_by].unique()

    lut = dict(zip(labels, "rbg"))

    print(lut)

    row_colors = [lut[x] for x in X[color_by]]

    print('row colors')
    print(row_colors)

    X.drop([color_by], axis = 'columns', inplace = True)

    def minmax_scaler(x):

        return (x - np.min(x)) /  (np.max(x) - np.min(x))

    norm = []

    for col in X:

        norm.append(minmax_scaler(X[col].values))
    
    #print(norm)
    X_norm = pd.DataFrame(norm).transpose()

    X_norm.columns = X.columns

    print(X_norm.shape)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40, 30))

    g = sns.clustermap(X_norm, cmap="mako", vmin=0, vmax=.4, row_colors = row_colors)

    #g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 30, rotation = 90)
    #g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 30, rotation = 0)

    # Draw the heatmap with the mask and correct aspect ratio
    return g