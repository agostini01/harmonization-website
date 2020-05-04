from django import forms

FEATURE_CHOICES = (
    ("1", "sepal_length"),
    ("2", "sepal_width"),
    ("3", "petal_length"),
    ("4", "petal_width"),
    ("5", "type"),
)


class FlowersForm(forms.Form):
    x_feature = forms.ChoiceField(choices=FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=FEATURE_CHOICES)
