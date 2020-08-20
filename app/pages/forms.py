from django import forms

from .choices.flowers import FLOWER_FEATURE_CHOICES
from .choices.unm import UNM_FEATURE_CHOICES, UNM_CATEGORICAL_CHOICES

""" (Name that will be send on the http request, name of the feature) """

PLOT_TYPES = (
    ("scatter_plot", "scatter_plot"),
    ("pair_plot", "pair_plot"),
    ("cat_plot", "cat_plot"),
    ("violin_cat_plot", "violin_cat_plot"),
    # TODO This is a 1D plot, and ignores y-feature
    ("histogram_plot", "histogram_plot"),
    ("linear_reg_plot", "linear_reg_plot"),
    ("linear_reg_with_color_plot", "linear_reg_with_color_plot"),
    ("linear_reg_detailed_plot", "linear_reg_detailed_plot")
)

DPI_CHOICES = (
    (100, "low_res"),
    (300, "high_res")
)

DATASET_CHOICES = (
    ("flowers_dataset", "flowers_dataset"),
    ("unm_dataset", "unm_dataset"),
    ("neu_dataset", "unm_dataset"),
    ("dar_dataset", "dar_dataset"),
    ("har_dataset", "har_dataset"),
)


class FlowersForm(forms.Form):

    def __init__(self, *args, **kwargs):
        super(FlowersForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = FLOWER_FEATURE_CHOICES[0][0]
        self.initial['y_feature'] = FLOWER_FEATURE_CHOICES[1][0]
        self.initial['color_by'] = FLOWER_FEATURE_CHOICES[4][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New Flowers Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[0][0]

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class UNMForm(forms.Form):
    """Form to select what features from the Raw UNM csv file to plot.

    This form must match the RawUNM model features. Restriction of choices here,
    will reflect there. 

    Other Files Involved:
        $PROJ_SOURCE/cohorts_proj/api/adapters/unm.py
        $PROJ_SOURCE/cohorts_proj/datasets/models.py
        $PROJ_SOURCE/cohorts_proj/datasets/mymodels/raw_unm.py
    """

    def __init__(self, *args, **kwargs):
        super(UNMForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = UNM_FEATURE_CHOICES[0][0]
        self.initial['y_feature'] = UNM_FEATURE_CHOICES[1][0]
        self.initial['color_by'] = UNM_CATEGORICAL_CHOICES[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New UNM Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[1][0]

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=UNM_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=UNM_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=UNM_CATEGORICAL_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())
