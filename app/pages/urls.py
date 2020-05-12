from django.urls import path

from .views import HomePageView, AboutPageView, GraphsPageView, UploadPageView


urlpatterns = [
    path('graphs/', GraphsPageView.as_view(), name='graphs'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('upload/', UploadPageView.as_view(), name='upload'),
    path('', HomePageView.as_view(), name='home'),
    path('graphs/getplot/', GraphsPageView.getPlot),
]
