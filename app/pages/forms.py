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
        ("arsenic_facet_continous", "arsenic_facet_continous"),
        ("covars_facet_continous", "covars_facet_continous"),
        ("covars_facet_categorical", "covars_facet_categorical")
    ),
    ),
    ('Categorical plots', (
        ("cat_plot", "cat_plot"),
        ("violin_cat_plot", "violin_cat_plot"),
    ),
    ),
    ('Data Summary', (
        ("continous_summary","continous_summary"),
        ("categorical_summary","categorical_summary"),
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
        ("bayesian_mixed_ml", "bayesian_mixed_ml"),
        ("binomial_bayesian_mixed_ml", "binomial_bayesian_mixed_ml"),
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

COVAR_CHOICES = (
    (False, False),
    (True, True)
)

DILUTION_CHOICES = (
    (False, False),
    (True, True)
)


COVAR_INDV_CHOICES = [
    ('age', 'age'),
    ('smoking', 'smoking'),
    ('babySex', 'babySex'),
    ('ga_collection', 'ga_collection'),
    ('education', 'education'),
    ('race', 'race'),
    ('BMI', 'BMI'),
    ('preg_complications', 'preg_complications'),
    ('folic_acid_supp', 'folic_acid_supp'),
    ('fish', 'fish'),
    ('parity','parity'),
]




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
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=FLOWER_FEATURE_CHOICES)
    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    time_period = forms.ChoiceField(choices=(('NA', 'NA')),
                                    widget=forms.HiddenInput())
   
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
        self.initial['x_feature'] = UNM_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = UNM_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = UNM_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_UNM_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New UNM Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[0][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=UNM_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=UNM_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=UNM_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_UNM_TIME_PERIOD,
                                    label='Time Period Filter')
 
    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class NEUForm(forms.Form):
    """Form to select what features from the Raw NEU csv file to plot.

    This form must match the RawNEU model features. Restriction of choices here,
    will reflect there. 

    Other Files Involved:
        $PROJ_SOURCE/cohorts_proj/api/adapters/neu.py
        $PROJ_SOURCE/cohorts_proj/datasets/models.py
        $PROJ_SOURCE/cohorts_proj/datasets/mymodels/raw_neu.py
    """

    def __init__(self, *args, **kwargs):
        super(NEUForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = NEU_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = NEU_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = NEU_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_NEU_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New NEU Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[1][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES)
    x_feature = forms.ChoiceField(choices=NEU_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=NEU_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=NEU_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_NEU_TIME_PERIOD,
                                    label='Time Period Filter')

    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class DARForm(forms.Form):
    """Form to select what features from the Raw DAR csv file to plot.

    This form must match the RawDAR model features. Restriction of choices here,
    will reflect there. 

    Other Files Involved:
        $PROJ_SOURCE/cohorts_proj/api/adapters/dar.py
        $PROJ_SOURCE/cohorts_proj/datasets/models.py
        $PROJ_SOURCE/cohorts_proj/datasets/mymodels/raw_dar.py
    """

    def __init__(self, *args, **kwargs):
        super(DARForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = DAR_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = DAR_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = DAR_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_DAR_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New Dartmouth Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[2][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES, label = 'Plot type / analysis')
    x_feature = forms.ChoiceField(choices=DAR_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=DAR_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=DAR_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_DAR_TIME_PERIOD,
                                    label='Time Period Filter')
  
    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class UNMNEUForm(forms.Form):
    """Form to select what features from the harmonized dataset to plot"""

    def __init__(self, *args, **kwargs):
        super(UNMNEUForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = UNMNEU_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = UNMNEU_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = UNMNEU_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_UNMNEU_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New UNM and NEU Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[3][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES, label = 'Plot type / analysis')
    x_feature = forms.ChoiceField(choices=UNMNEU_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=UNMNEU_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=UNMNEU_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_UNMNEU_TIME_PERIOD,
                                    label='Time Period Filter')
  
    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class NEUDARForm(forms.Form):
    """Form to select what features from the harmonized dataset to plot"""

    def __init__(self, *args, **kwargs):
        super(NEUDARForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = NEUDAR_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = NEUDAR_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = NEUDAR_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_NEUDAR_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New NEU and DAR Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[4][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES, label = 'Plot type / analysis')
    x_feature = forms.ChoiceField(choices=NEUDAR_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=NEUDAR_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=NEUDAR_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_NEUDAR_TIME_PERIOD,
                                    label='Time Period Filter')

    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class DARUNMForm(forms.Form):
    """Form to select what features from the harmonized dataset to plot"""

    def __init__(self, *args, **kwargs):
        super(DARUNMForm, self).__init__(*args, **kwargs)
        self.initial['plot_type'] = PLOT_TYPES[0][0]
        self.initial['x_feature'] = DARUNM_FEATURE_CHOICES[0][1][0][0]
        self.initial['y_feature'] = DARUNM_FEATURE_CHOICES[0][1][1][0]
        self.initial['color_by'] = DARUNM_CATEGORICAL_CHOICES[0][1][0][0]
        self.initial['time_period'] = CAT_DARUNM_TIME_PERIOD[0][0]
        self.initial['fig_dpi'] = DPI_CHOICES[0][0]
        self.initial['plot_name'] = 'New DAR and UNM Plot'
        self.initial['dataset_type'] = DATASET_CHOICES[5][0]
        self.initial['include_covars'] = False

    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES, label = 'Plot type / analysis')
    x_feature = forms.ChoiceField(choices=DARUNM_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=DARUNM_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=DARUNM_CATEGORICAL_CHOICES)
    time_period = forms.ChoiceField(choices=CAT_DARUNM_TIME_PERIOD,
                                    label='Time Period Filter')

    covar_choices = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )
    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())


class HARForm(forms.Form):
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
        self.initial['adjust_dilution'] = False


    plot_name = forms.CharField(max_length=100,
                                help_text="Type the name of your next plot.")
    plot_type = forms.ChoiceField(choices=PLOT_TYPES, label = 'Plot type / analysis')
    x_feature = forms.ChoiceField(choices=HAR_FEATURE_CHOICES)
    y_feature = forms.ChoiceField(choices=HAR_FEATURE_CHOICES)
    color_by = forms.ChoiceField(choices=HAR_CATEGORICAL_CHOICES)

    time_period = forms.ChoiceField(choices=CAT_HAR_TIME_PERIOD,
                                    label='Time Period Filter')



    covar_choices = forms.MultipleChoiceField(
        label='Covariate Selection',
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=COVAR_INDV_CHOICES,
    )

    adjust_dilution = forms.ChoiceField(choices=DILUTION_CHOICES)

    fig_dpi = forms.ChoiceField(choices=DPI_CHOICES,
                                help_text="low_res=100dpi, high_res=300dpi.")

    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES,
                                     widget=forms.HiddenInput())

