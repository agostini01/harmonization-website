from django.contrib import admin

# Raw dataset models
from datasets.models import RawFlower, RawUNM, RawDAR

# Register your models here.
admin.site.register(RawFlower)
admin.site.register(RawUNM)
admin.site.register(RawDAR)
