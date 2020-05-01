from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView


urlpatterns = [
    path('graphs/', GraphsPageView.as_view(), name='graphs'), 
    path('about/', AboutPageView.as_view(), name='about'), 
    path('', HomePageView.as_view(), name='home'),
    path('getsbdata/', GraphsPageView.getSBData),
]