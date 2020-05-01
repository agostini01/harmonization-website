from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse

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


class GraphsPageView(TemplateView): 
    template_name = 'graphs/graphs_base.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

    @classmethod
    def getSBData(cls, request):
        #gr=sb.factorplot(x='Survived', hue='Sex', data=df, col='Pclass', kind='count')
        gr=sns.pairplot(iris,hue="type",height=3)
        response = HttpResponse(content_type="image/jpg")
        gr.savefig(response, format="jpg")

        return response
