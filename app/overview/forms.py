from django import forms

from .choices.flowers import FLOWER_FEATURE_CHOICES
from .choices.neu import NEU_FEATURE_CHOICES, NEU_CATEGORICAL_CHOICES, CAT_NEU_TIME_PERIOD
from .choices.unm import UNM_FEATURE_CHOICES, UNM_CATEGORICAL_CHOICES, CAT_UNM_TIME_PERIOD
from .choices.dar import DAR_FEATURE_CHOICES, DAR_CATEGORICAL_CHOICES, CAT_DAR_TIME_PERIOD

from .choices.unmneu import UNMNEU_FEATURE_CHOICES, UNMNEU_CATEGORICAL_CHOICES, CAT_UNMNEU_TIME_PERIOD
from .choices.neudar import NEUDAR_FEATURE_CHOICES, NEUDAR_CATEGORICAL_CHOICES, CAT_NEUDAR_TIME_PERIOD
from .choices.darunm import DARUNM_FEATURE_CHOICES, DARUNM_CATEGORICAL_CHOICES, CAT_DARUNM_TIME_PERIOD

from .choices.har import HAR_FEATURE_CHOICES, HAR_CATEGORICAL_CHOICES, CAT_HAR_TIME_PERIOD

""" (Name that will be send on the http request, name of the feature) """

PLOT_TYPES = [
    ('2D plots', (
        ("individual_scatter_plot", "individual_scatter_plot"),
        ("scatter_plot", "scatter_plot"),
        ("pair_plot", "pair_plot"),
        ("corr_plot", "corr_plot"),
        ("clustermap","clustermap"),
        ("analysis", "analysis"),
        ("arsenic_facet_continous", "arsenic_facet_continous"),

        ("covars_facet_continous", "covars_facet_continous"),
        ("covars_facet_categorical", "covars_facet_categorical"),
         ("custom_facet_LM_plot", "custom_facet_LM_plot"),
    ),
    ),
    ('Categorical plots', (
        ("cat_plot", "cat_plot"),
        ("violin_cat_plot", "violin_cat_plot"),
    ),
    ),
    ('1D plots', (
        ("histogram_plot", "histogram_plot"),
         ("kde_plot", "kde_plot")
    ),
    ),
    ('Regressions', (
        ("linear_reg_plot", "linear_reg_plot"),
        ("linear_reg_with_color_plot", "linear_reg_with_color_plot"),
        ("linear_reg_detailed_plot", "linear_reg_detailed_plot"),
        ("linear_mixed_ml_summary", "linear_mixed_ml_summary"),
        ("binomial_mixed_ml_summary", "binomial_mixed_ml_summary"),
        ("logistic_regression", "logistic_regression"),
    ),
    )
]

DPI_CHOICES = (
    (100, "low_res"),
    (300, "high_res")
)

DATASET_CHOICES = (
    ("unm_dataset", "unm_dataset"),
    ("neu_dataset", "unm_dataset"),
    ("dar_dataset", "dar_dataset"),
    ("unmneu_dataset", "unmneu_dataset"),
    ("neudar_dataset", "unmdar_dataset"),
    ("darunm_dataset", "darunm_dataset"),
    ("har_dataset", "har_dataset"),
)


class FlowersForm(forms.Form):

    def __init__(self, *args, **kwargs):
        super(FlowersForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = FLOWER_FEATURE_CHOICES[0][0]
        self.initial['y_feature'] = FLOWER_FEATURE_CHOICES[1][0]
        self.initial['color_by'] = FLOWER_FEATURE_CHOICES[4][0]
        self.initial['time_period'] = 'na'
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New Flowers Plot'
        self.initial['dataset_type'] = 'flowers_dataset'

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    time_period = forms.ChoiceField(choices=(('NA', 'NA')),
                                    widget=forms.HiddenInput())
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class HAROverviewForm(forms.Form):
    """Form to select what features from the harmonized dataset to plot"""

    def __init__(self, *args, **kwargs):
        super(HARForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = HAR_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = HAR_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = HAR_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_HAR_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New Harmonized Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[6][0]

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=HAR_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=HAR_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=HAR_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_HAR_TIME_PERIOD,
                                    label='Time Period Filter')
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())
