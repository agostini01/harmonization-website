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

class RawFlower(models.Model):

    # Just a number for the sample
    PIN_ID = models.CharField(max_length=100)

    # Result: value in cm
    sepal_length = models.FloatField()
    sepal_width = models.FloatField()
    petal_length = models.FloatField()
    petal_width = models.FloatField()

    # The type of flower
    type_field = FlowerTypeField(max_length=100, choices=CAT_FLO_TYPE)
