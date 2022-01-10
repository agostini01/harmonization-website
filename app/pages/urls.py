from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView, load_dataset
from .views import GraphsFlowersPagesView
from .views import GraphsUNMPagesView
from .views import GraphsNEUPagesView
from .views import GraphsNEUtestPagesView
from .views import GraphsDARPagesView
from .views import GraphsUNMNEUPagesView
from .views import GraphsNEUDARPagesView
from .views import GraphsDARUNMPagesView
from .views import GraphsHARPagesView
from .views import GraphsHAROverviewpagesView
from .views import DictionariesAllPagesView
from .views import DictionariesPageView
from .views import DictionarytestPagesView

from .views import load_dataset

urlpatterns = [
    #path('graphs/flowers/', GraphsFlowersPagesView.as_view(), name='graphs-flowers'),
    #path('graphs/unm/', GraphsUNMPagesView.as_view(), name='graphs-unm'),
    #path('graphs/neu/', GraphsNEUPagesView.as_view(), name='graphs-neu'),
    path('graphs/neu/', GraphsNEUtestPagesView.as_view(), name='graphs-neu'),
    #path('graphs/dar/', GraphsDARPagesView.as_view(), name='graphs-dar'),
    #path('graphs/unm-neu/', GraphsUNMNEUPagesView.as_view(), name='graphs-unm-neu'),
    #path('graphs/neu-dar/', GraphsNEUDARPagesView.as_view(), name='graphs-neu-dar'),
    #path('graphs/dar-unm/', GraphsDARUNMPagesView.as_view(), name='graphs-dar-unm'),
    #path('graphs/har/', GraphsHARPagesView.as_view(), name='graphs-har'),
    path('overview/haroverview/',GraphsHAROverviewpagesView.as_view(), name = 'graphs-haroverview' ),
    path('graphs/', GraphsHARPagesView.as_view(), name='graphs'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('dictionaries/', DictionarytestPagesView.as_view(), name='dictionaries'),
    #path('dictionaries/overview/', GraphsHAROverviewpagesView.getOverviewPlot),
    path('', HomePageView.as_view(), name='home'),
    path('load-datasets', load_dataset, name='ajax_load_datasets'),

    # TODO - Remove once all graph logic gets ported
    #path('graphs/getplot/', GraphsPageView.getPlot),
    #path('graphs/api/getdict/', )
    path('graphs/api/getDictInfo', GraphsPageView.getDictInfo),
    path('graphs/api/getplot/', GraphsPageView.getApiPlot),
    path('graphs/api/getinfo/', GraphsPageView.getApiInfo),
]
