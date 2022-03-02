from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect
from django.core import serializers

from .forms import DictionaryForm, FlowersForm, UNMForm, NEUForm, DARForm, HARForm
from .forms import NEUForm_test
#from .forms import HAROverviewForm
from .forms import UNMNEUForm, NEUDARForm, DARUNMForm
from .validation import checkFormRequest, getErrorImage
from django.shortcuts import render, redirect

from .choices.flowers import FLOWER_FEATURE_CHOICES
from .choices.neu import NEU_FEATURE_CHOICES, NEU_CATEGORICAL_CHOICES, CAT_NEU_TIME_PERIOD
from .choices.unm import UNM_FEATURE_CHOICES, UNM_CATEGORICAL_CHOICES, CAT_UNM_TIME_PERIOD
from .choices.dar import DAR_FEATURE_CHOICES, DAR_CATEGORICAL_CHOICES, CAT_DAR_TIME_PERIOD

from .choices.unmneu import UNMNEU_FEATURE_CHOICES, UNMNEU_CATEGORICAL_CHOICES, CAT_UNMNEU_TIME_PERIOD
from .choices.neudar import NEUDAR_FEATURE_CHOICES, NEUDAR_CATEGORICAL_CHOICES, CAT_NEUDAR_TIME_PERIOD
from .choices.darunm import DARUNM_FEATURE_CHOICES, DARUNM_CATEGORICAL_CHOICES, CAT_DARUNM_TIME_PERIOD

from .choices.har import HAR_FEATURE_CHOICES, HAR_CATEGORICAL_CHOICES, CAT_HAR_TIME_PERIOD

import requests
import json
from django.views.decorators.csrf import csrf_exempt



def load_dict(request):
    feature_id = request.GET.get('dataset')

@csrf_exempt
def postrec(request):
    url = "http://api:8887/query/dataset-upload/"

    data = request.POST.get('rec').split('|')
  
    payload = {'id': data[0],
               'var1': data[1],
               'var2': data[2],
               'uploader_email': data[1],
               'dataset_type': 'harm_save'}

    headers = {}

    response = requests.request(
        "POST", url, headers=headers, data=payload)

    
    #return response
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/'))

def getrec(request):
    
    context = {'rec':1}
    plot_type = 'recrequest'
    x_feature = request.GET.get('shoppingBasket')
    y_feature = ''
    color_by = ''
    time_period = 1
    fig_dpi = 20
    dataset_type = ''
    covar_choices = ''
    adjust_dilution = ''

    url = "http://api:8887/query/get-dict/"

    payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'recrequest',
                   'dataset_type': dataset_type,
                   'adjust_dilution': adjust_dilution,
                   'covar_choices': covar_choices}
                
    files = []
    headers = {}

    requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)
    
    print(type(requests_response.content))
    my_json = requests_response.content.decode('utf8').replace("'", '"').replace("None", "212").replace('nan','0').replace('["','').replace('"]','')

    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(my_json)

    context = dict({'items': data})

    return render(request, 'dictionaries/return_rec.html',context)
def code_analytes(feature_choices):

    analytes =[{'short': x, 'long': y} for x, y in feature_choices[0][1]]
    categorical =[{'short': x, 'long': y} for x, y in feature_choices[1][1]]
    outcomes =[{'short': x, 'long': y} for x, y in feature_choices[2][1]]
    return analytes, categorical, outcomes

def load_dataset(request):
    feature_id = request.GET.get('dataset')
    #cities = City.objects.filter(country_id=country_id).order_by('name')
    
    if feature_id == 'unm_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(UNM_FEATURE_CHOICES)
    if feature_id == 'dar_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(DAR_FEATURE_CHOICES)   
    if feature_id == 'neu_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(NEUDAR_FEATURE_CHOICES)
    if feature_id == 'unmneu_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(UNMNEU_FEATURE_CHOICES)
    if feature_id == 'darunm_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(DARUNM_FEATURE_CHOICES)   
    if feature_id == 'neudar_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(NEUDAR_FEATURE_CHOICES)
    if feature_id == 'har_dataset':
        x_features = UNM_FEATURE_CHOICES
        analytes ,categorical, outcomes = code_analytes(HAR_FEATURE_CHOICES)

      
    return render(request, 'graphs/load_datasets.html', {'analytes': analytes, 'categorical': categorical, 'outcomes': outcomes})

class HomePageView(TemplateView):
    template_name = 'home.html'


class AboutPageView(LoginRequiredMixin, TemplateView):
    template_name = 'about.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

class AboutPageView2(LoginRequiredMixin, TemplateView):
    template_name = 'about.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'


class GraphsPageView(LoginRequiredMixin, FormView):
    template_name = 'graphs/graphs_base.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

    form_class = FlowersForm

    @classmethod
    def getApiInfo(cls, request):

        """This function requets graphs through the API container."""
        print(request)
        err = checkFormRequest(request)

        if (err[0] != 0):
            return getErrorImage(err)

        plot_type = request.GET.get('plot_type')
        x_feature = request.GET.get('x_feature')
        y_feature = request.GET.get('y_feature')
        color_by = request.GET.get('color_by')
        time_period = int(request.GET.get('time_period'))
        fig_dpi = int(request.GET.get('fig_dpi'))
        dataset_type = request.GET.get('dataset_type')
        covar_choices = request.GET.get('covar_choices')
        adjust_dilution = request.GET.get('adjust_dilution')

        url = "http://api:8887/query/get-info/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'test',
                   'dataset_type': dataset_type,
                   'adjust_dilution': adjust_dilution,
                   'covar_choices': covar_choices}
                
        files = []
        headers = {}

        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        django_response = HttpResponse(
            content=requests_response.content,
            status=requests_response.status_code,
            content_type=requests_response.headers['Content-Type']
        )

        print(django_response)
        return django_response

    @classmethod
    def getDictInfo(cls, request):

        """This function requets graphs through the API container."""
        print(request)
        err = checkFormRequest(request)

        if (err[0] != 0):
            return getErrorImage(err)

        
        print(request.GET)
        plot_type = request.GET.get('plot_name')
        x_feature = ''
        y_feature = ''
        color_by = ''
        time_period = 1
        fig_dpi = 100
        dataset_type = ''
        covar_choices = ''
        adjust_dilution = ''

        print(plot_type)

        url = "http://api:8887/query/get-dict/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'test',
                   'dataset_type': dataset_type,
                   'adjust_dilution': adjust_dilution,
                   'covar_choices': covar_choices}
                
        files = []
        headers = {}

        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        django_response = HttpResponse(
            content=requests_response.content,
            status=requests_response.status_code,
            content_type=requests_response.headers['Content-Type']
        )

        print(django_response)
        return django_response



    @classmethod
    def getApiPlot(cls, request):
        """This function requets graphs through the API container."""
        print(request)
        err = checkFormRequest(request)

        if (err[0] != 0):
            return getErrorImage(err)

        plot_type = request.GET.get('plot_type')
        x_feature = request.GET.get('x_feature')
        y_feature = request.GET.get('y_feature')
        color_by = request.GET.get('color_by')
        time_period = int(request.GET.get('time_period'))
        fig_dpi = int(request.GET.get('fig_dpi'))
        dataset_type = request.GET.get('dataset_type')

        url = "http://api:8887/query/get-plot/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'test',
                   'dataset_type': dataset_type}
                   
        files = []
        headers = {}

        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        django_response = HttpResponse(
            content=requests_response.content,
            status=requests_response.status_code,
            content_type=requests_response.headers['Content-Type']
        )
        
        print(django_response)
        return django_response

    def get_context_data(self, **kwargs):
        context = super(GraphsPageView, self).get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        return super(GraphsPageView, self).form_valid(form)

class DictionariesPageView(LoginRequiredMixin, FormView):
    template_name = 'dictionaries/dictionaries_base.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'

    form_class = DictionaryForm

    #@classmethod
    def get(cls, request):
        
        print('request-->', request)

        """This function requets graphs through the API container."""
        print(request)
        err = checkFormRequest(request)

        if (err[0] != 0):
            return getErrorImage(err)

        plot_type = request.GET.get('plot_name')
        x_feature = ''
        y_feature = ''
        color_by = ''
        time_period = 100
        fig_dpi = 1
        dataset_type = ''
        covar_choices = ''
        adjust_dilution = ''

        if plot_type is None:
            plot_type = 'test'
        

        url = "http://api:8888/query/get-dict/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': plot_type,
                   'dataset_type': dataset_type,
                   'adjust_dilution': adjust_dilution,
                   'covar_choices': covar_choices}
                
        files = []
        headers = {}

        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

    
        #print(requests_response.content)
        my_json = requests_response.content.decode('utf8').replace("'", '"').replace("None", "212")

        # Decode UTF-8 bytes to Unicode, and convert single quotes 
        # to double quotes to make it valid JSON
        #my_json = my_bytes_value.decode('utf8').replace("'", '"')
        print(my_json)
        print('- ' * 20)

        # Load the JSON to a Python list & dump it back out as formatted JSON
        data = json.loads(my_json)
    

        
        context = dict({'items': data})

        return render(request, 'dictionaries/dictionaries_base.html',context)

        #return HttpResponse(template.render(context, request))




class DictionariesAllPagesView(DictionariesPageView):
    form_class = FlowersForm

class GraphsFlowersPagesView(GraphsPageView):
    form_class = FlowersForm


class GraphsUNMPagesView(GraphsPageView):
    form_class = UNMForm


class GraphsNEUPagesView(GraphsPageView):
    form_class = NEUForm

class GraphsNEUtestPagesView(GraphsPageView):
    form_class = NEUForm_test

class DictionarytestPagesView(DictionariesPageView):
    form_class = DictionaryForm

class GraphsDARPagesView(GraphsPageView):
    form_class = DARForm


class GraphsUNMNEUPagesView(GraphsPageView):
    form_class = UNMNEUForm


class GraphsNEUDARPagesView(GraphsPageView):
    form_class = NEUDARForm


class GraphsDARUNMPagesView(GraphsPageView):
    form_class = DARUNMForm


class GraphsHARPagesView(GraphsPageView):
    form_class = HARForm

class GraphsHAROverviewpagesView(GraphsPageView):
    form_class = HARForm
