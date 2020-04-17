from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from upload.views import image_upload

urlpatterns = [
    # Django Admin
    path("admin/", admin.site.urls),
    #path("", image_upload, name="upload"),

    # User management
    path('accounts/', include('allauth.urls')),

    # Local Apps
    path('', include('pages.urls')),
]

if bool(settings.DEBUG):
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
