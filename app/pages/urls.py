from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView


urlpatterns = [
    path('graphs/flowers/', GraphsFlowersPagesView.as_view(), name='graphs-flowers'),
    path('graphs/', GraphsPageView.as_view(), name='graphs'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('', HomePageView.as_view(), name='home'),

    # TODO - Remove once all graph logic gets ported
    #path('graphs/getplot/', GraphsPageView.getPlot),
    path('graphs/api/getplot/', GraphsPageView.getApiPlot),
]
