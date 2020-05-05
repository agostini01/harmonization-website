from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

from .forms import FlowersForm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
iris = pd.read_csv("staticfiles/datasets/iris.csv", names=col)
iris_setosa = iris.loc[iris["type"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["type"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["type"] == "Iris-versicolor"]


class HomePageView(TemplateView):
    template_name = 'home.html'


class AboutPageView(LoginRequiredMixin, TemplateView):
    template_name = 'about.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'


class GraphsPageView(FormView):
    template_name = 'graphs/graphs_base.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

    form_class = FlowersForm

    @classmethod
    def getScatterPlot(cls, x_feature, y_feature, color_by):
        gr = sns.scatterplot(
            data=iris, x=x_feature, y=y_feature, hue=color_by)

        return gr.figure 
 
    @classmethod
    def getPairPlot(cls, x_feature, y_feature, color_by):
        gr = sns.pairplot(
            iris, vars=[x_feature, y_feature], hue=color_by, height=3)

        return gr
    
    @classmethod
    def getCatPlot(cls, x_feature, y_feature, color_by):
        gr = sns.catplot(data=iris, x=x_feature,
                         y=y_feature, hue=color_by)

        return gr
    
    @classmethod
    def getViolinCatPlot(cls, x_feature, y_feature, color_by):
        gr = sns.catplot(data=iris, x=x_feature,
                         y=y_feature, hue=color_by, kind="violin")

        return gr

    @classmethod
    def getPlot(cls, request):
        plot_type = request.GET.get('plot_type')
        x_feature = request.GET.get('x_feature')
        y_feature = request.GET.get('y_feature')
        color_by = request.GET.get('color_by')
        fig_dpi = int(request.GET.get('fig_dpi'))

        t = plot_type
        gr = None

        plt.clf()
        if (t == 'scatter_plot'):
            gr = cls.getScatterPlot(x_feature, y_feature, color_by)

        if (t == 'pair_plot'):
            gr = cls.getPairPlot(x_feature, y_feature, color_by)
        
        if (t == 'cat_plot'):
            gr = cls.getCatPlot(x_feature, y_feature, color_by)
        
        if (t == 'violin_cat_plot'):
            gr = cls.getViolinCatPlot(x_feature, y_feature, color_by)

        #Add histogram

        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg", dpi=fig_dpi)

        return response


    def get_context_data(self, **kwargs):
        context = super(GraphsPageView, self).get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        return super(GraphsPageView, self).form_valid(form)
