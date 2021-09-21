from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView
from .views import GraphsFlowersPagesView
from .views import GraphsUNMPagesView
from .views import GraphsNEUPagesView
from .views import GraphsDARPagesView
from .views import GraphsUNMNEUPagesView
from .views import GraphsNEUDARPagesView
from .views import GraphsDARUNMPagesView
from .views import GraphsHARPagesView
from .views import GraphsHAROverviewpagesView

urlpatterns = [
    path('',GraphsHAROverviewpagesView.as_view(), name = 'haroverview' ),
    #path('graphs/', GraphsHARPagesView.as_view(), name='graphs'),
    #path('about/', AboutPageView.as_view(), name='about'),
    path('', HomePageView.as_view(), name='home'),

    # TODO - Remove once all graph logic gets ported
    #path('graphs/getplot/', GraphsPageView.getPlot),
    path('graphs/api/getplot/', GraphsPageView.getApiPlot),
]
