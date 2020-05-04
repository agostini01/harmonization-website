from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

from .forms import FlowersForm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

col=['sepal_length','sepal_width','petal_length','petal_width','type']
iris=pd.read_csv("staticfiles/datasets/iris.csv",names=col)
iris_setosa=iris.loc[iris["type"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["type"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["type"]=="Iris-versicolor"]


class HomePageView(TemplateView):
    template_name = 'home.html'


class AboutPageView(LoginRequiredMixin,TemplateView): 
    template_name = 'about.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'


class GraphsPageView(FormView): 
    template_name = 'graphs/graphs_base.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

    form_class = FlowersForm

    @classmethod
    def getPairPlot(cls, request):
        print("HERE!")
        #gr=sb.factorplot(x='Survived', hue='Sex', data=df, col='Pclass', kind='count')
        gr=sns.pairplot(iris,hue="type",height=3)
        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg")

        return response

    def get_context_data(self, **kwargs):
        context = super(GraphsPageView, self).get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        return super(GraphsPageView, self).form_valid(form)
