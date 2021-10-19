from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect
from .forms import FlowersForm, HAROverviewForm
#from .forms import HAROverviewForm
from .validation import checkFormRequest, getErrorImage
from django.shortcuts import render
import json
import requests


class HomePageView(TemplateView):
    template_name = 'home.html'


class AboutPageView(LoginRequiredMixin, TemplateView):
    template_name = 'about.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'


class GraphsHAROverviewpagesView(LoginRequiredMixin, TemplateView):
    #template_name = 'overview.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'
    
    @classmethod
    def get(self, request):
        """This function requets graphs through the API container."""
        plot_type = 'overview_report'
        x_feature = 'USB'
        y_feature = 'UTAS'
        color_by = 'Outcome'
        time_period = '93'
        fig_dpi = '100'
        dataset_type = 'neu_dataset'
        covar_choices = ''
        adjust_dilution = 'False'

        url = "http://api:8887/query/get-info/"
                
        files = []
        headers = {}

        payload = {'plot_type': plot_type,
                    'x_feature': x_feature,
                    'y_feature': y_feature,
                    'color_by': color_by,
                    'time_period': time_period,
                    'fig_dpi': fig_dpi,
                    'plot_name': 'test',
                    'dataset_type': dataset_type,
                    'covar_choices': covar_choices,
                    'adjust_dilution': adjust_dilution
                    }
                    
        files = []
        headers = {}


        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        django_response = HttpResponse(
            content=requests_response.content,
            status=requests_response.status_code,
            content_type=requests_response.headers['Content-Type']
        )
        
        json_records= django_response.content

        # parsing the DataFrame in json format. 
        #json_records = df.reset_index().to_json(orient ='records') 
        data = [] 
        data = json.loads(json_records) 
        context = {'d': data} 

        template_name="overview.html"

        return render(request, template_name, context)

    @classmethod
    def getOverviewPlot(self, request):
        """This function requets graphs through the API container."""
        plot_type = 'overview_plot'
        x_feature = 'USB'
        y_feature = 'UTAS'
        color_by = 'Outcome'
        time_period = '93'
        fig_dpi = '100'
        dataset_type = 'neu_dataset'
        covar_choices = ''

        url = "http://api:8887/query/get-plot/"
                
        files = []
        headers = {}

        payload = {'plot_type': plot_type,
                    'x_feature': x_feature,
                    'y_feature': y_feature,
                    'color_by': color_by,
                    'time_period': time_period,
                    'fig_dpi': fig_dpi,
                    'plot_name': 'test',
                    'dataset_type': dataset_type
                    }
                    
        files = []
        headers = {}


        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        print('**********')
        print(requests_response)
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