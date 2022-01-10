from django.db import models

CAT_FLO_TYPE = (
    ('iris-setosa', 'Iris-setosa'),
    ('iris-virginica', 'Iris-virginica'),
    ('iris-versicolor', 'Iris-versicolor'),
)

class FlowerTypeField(models.CharField):
    def __init__(self, *args, **kwargs):
        super(FlowerTypeField, self).__init__(*args, **kwargs)

    def get_prep_value(self, value):
        return str(value).lower()

class RawDictionary(models.Model):

    cohort = models.CharField(max_length=1000)
    var_name = models.CharField(max_length=1000)
    form_name = models.CharField(max_length=1000, null = True, blank=True)
    section_name = models.CharField(max_length=1000, null = True, blank=True)
    field_type = models.CharField(max_length=1000, null = True, blank=True)
    field_label = models.CharField(max_length =1000, null = True, blank=True)
    field_choices = models.CharField(max_length= 1000, null = True, blank=True)
    field_min = models.CharField(max_length= 1000, null = True, blank=True)
    field_max = models.CharField(max_length= 1000, null = True, blank=True)

    # The type of flower
    #flower_type = FlowerTypeField(max_length=100, choices=CAT_FLO_TYPE)
