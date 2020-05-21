from django.contrib import admin

# Raw dataset models
from datasets.models import RawFlower, RawUNM

# Register your models here.
admin.site.register(RawFlower)
admin.site.register(RawUNM)
