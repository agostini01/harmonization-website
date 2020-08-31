import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats

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
    gr = sns.catplot(data=data, x=x_feature,
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
    gr = sns.lmplot(data=data, x=x_feature,
                    y=y_feature, hue=color_by, legend_out=True)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=gr.axes.flat[0].get_lines()[0].get_xdata(),
        y=gr.axes.flat[0].get_lines()[0].get_ydata())
    reg_info0 = "f(x)={:.3f}x + {:.3f}".format(
        slope, intercept)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=gr.axes.flat[0].get_lines()[1].get_xdata(),
        y=gr.axes.flat[0].get_lines()[1].get_ydata())
    reg_info1 = "g(x)={:.3f}x + {:.3f}".format(
        slope, intercept)

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