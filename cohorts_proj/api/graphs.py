import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

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

def getRegPlot(data, x_feature, y_feature, color_by):
    fig, _ = plt.subplots(1, 1, figsize=(5, 5))
    gr = sns.regplot(data=data, x=x_feature,
                     y=y_feature)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=gr.get_lines()[0].get_xdata(),
        y=gr.get_lines()[0].get_ydata())
    reg_info = "f(x)={:.3f}x + {:.3f}".format(
        slope, intercept)

    gr.set_title(reg_info)

    return gr.figure


def getRegColorPlot(data, x_feature, y_feature, color_by):

    filtered_df = data[data[[x_feature, y_feature]].notnull().all(1)]
    color_by_options = filtered_df[color_by].unique()

    reg_info0 = ''
    reg_info1 = ''
    gr = sns.lmplot(data=data, x=x_feature,
                    y=y_feature, hue=color_by, legend_out=True)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=gr.axes.flat[0].get_lines()[0].get_xdata(),
        y=gr.axes.flat[0].get_lines()[0].get_ydata())
    reg_info0 = "f(x)={:.3f}x + {:.3f}".format(
        slope, intercept)

    # for x in xrange(len(color_by_options)):

    if (len(color_by_options) > 1):
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x=gr.axes.flat[0].get_lines()[1].get_xdata(),
                y=gr.axes.flat[0].get_lines()[1].get_ydata())
            reg_info1 = "g(x)={:.3f}x + {:.3f}".format(
                slope, intercept)

        except Exception as exc:
            print('Error: We need 2 points to create a line...')
            print(traceback.format_exc())
            print(exc)

    reg_info = "{}  |  {}".format(reg_info0, reg_info1)

    gr.fig.suptitle(reg_info)

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


