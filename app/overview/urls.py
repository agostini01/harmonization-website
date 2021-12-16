from django.urls import path

from .views import HomePageView, AboutPageView

from .views import GraphsHAROverviewpagesView

urlpatterns = [
    path('',GraphsHAROverviewpagesView.as_view(), name = 'haroverview' ),
    path('overviewplot/',GraphsHAROverviewpagesView.getOverviewPlot, name = 'overviewplot' ),
    path('overviewdownload/<str:id1>/<str:id2>',GraphsHAROverviewpagesView.getOverviewDownload, name = 'overviewdownload' ),

    
    #path('graphs/', GraphsHARPagesView.as_view(), name='graphs'),
    #path('about/', AboutPageView.as_view(), name='about'),
    path('', HomePageView.as_view(), name='home'),

    # TODO - Remove once all graph logic gets ported
    #path('graphs/getplot/', GraphsPageView.getPlot),
    #path('graphs/api/getplot/', GraphsPageView.getApiPlot),
    #path('graphs/api/getinfo/', GraphsPageView.getApiInfo),
    #path('overview/api/getoverview/', GraphsHAROverviewpagesView.getApiInfo),
]
