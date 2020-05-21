from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView
from .views import GraphsFlowersPagesView
from .views import GraphsUNMPagesView


urlpatterns = [
    path('graphs/flowers/', GraphsFlowersPagesView.as_view(), name='graphs-flowers'),
    path('graphs/unm/', GraphsUNMPagesView.as_view(), name='graphs-unm'),
    path('graphs/', GraphsPageView.as_view(), name='graphs'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('', HomePageView.as_view(), name='home'),

    # TODO - Remove once all graph logic gets ported
    #path('graphs/getplot/', GraphsPageView.getPlot),
    path('graphs/api/getplot/', GraphsPageView.getApiPlot),
]
