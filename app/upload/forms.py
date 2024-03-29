from django import forms

from .validators import validate_csv


DATASET_CHOICES = (
    # ("flowers_dataset", "flowers_dataset"),
    ("UNM_dataset", "UNM_dataset"),
    ("NEU_dataset", "NEU_dataset"),
    ("Dartmouth_dataset", "Dartmouth_dataset"),
    ("NHANES_bio", "NHANES_bio"),
    ("NHANES_llod", "NHANES_llod"),
    ("dictionary", "dictionary"),
    ("csv_only", "csv_only")
)


class CSVFileField(forms.FileField):
    allow_empty_file = False

    default_validators = [validate_csv]


class UploadFileForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)

    uploader_name = forms.CharField(max_length=100,
                                    help_text="Scientist uploading this dataset.")
    uploader_email = forms.EmailField(max_length=100,
                                      help_text='A valid email address, please.')
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES)
    #dataset_type = forms.CharFeild(max_length=100)
    dataset_file = CSVFileField(
        help_text="Only csv files are accepted. "
        "The csv file MUST have the correct header.")
