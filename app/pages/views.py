from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect

from .forms import FlowersForm

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
    def getApiPlot(cls, request):
        """This function requets graphs through the API container."""

        plot_type = request.GET.get('plot_type')
        x_feature = request.GET.get('x_feature')
        y_feature = request.GET.get('y_feature')
        color_by = request.GET.get('color_by')
        fig_dpi = int(request.GET.get('fig_dpi'))

        url = "http://api:8888/query/get-plot/"

        payload = {'plot_type': plot_type,
                   'x_feature': x_feature,
                   'y_feature': y_feature,
                   'color_by': color_by,
                   'fig_dpi': fig_dpi,
                   'plot_name': 'test'}
        files = []
        headers = {}

        requests_response = requests.request(
            "GET", url, headers=headers, data=payload, files=files)

        django_response = HttpResponse(
            content=requests_response.content,
            status=requests_response.status_code,
            content_type=requests_response.headers['Content-Type']
        )
        return django_response

    def get_context_data(self, **kwargs):
        context = super(GraphsPageView, self).get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        return super(GraphsPageView, self).form_valid(form)
