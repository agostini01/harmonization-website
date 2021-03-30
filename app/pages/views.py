from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect

from .forms import FlowersForm, UNMForm, NEUForm, DARForm, HARForm
from .forms import UNMNEUForm, NEUDARForm, DARUNMForm
from .validation import checkFormRequest, getErrorImage

import requests


class HomePageView(TemplateView):
    template_name = 'home.html'


class AboutPageView(LoginRequiredMixin, TemplateView):
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

        url = "http://api:8888/query/get-info/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'time_period': time_period,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'test',
                   'dataset_type': dataset_type,
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

        url = "http://api:8888/query/get-plot/"

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


class GraphsFlowersPagesView(GraphsPageView):
    form_class = FlowersForm


class GraphsUNMPagesView(GraphsPageView):
    form_class = UNMForm


class GraphsNEUPagesView(GraphsPageView):
    form_class = NEUForm


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
