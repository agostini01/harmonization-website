from django.core.exceptions import ValidationError

from django import forms

# import csv

# Custom Form Validators


def validate_csv(value):

    if not value.name.endswith('.csv'):
        raise ValidationError("Upload a valid csv file. "
                              "The file you uploaded was either: not a "
                              "csv file or a corrupted csv file.", code=500)

    # TODO
    # This is commented out because it closes the file. Must find a way to
    # leave it open
    # with value.open(mode='rb') as csvfile:

    #     try:
    #         csvreader = csv.reader(csvfile)
    #         # Do whatever checks you want here
    #         # Raise ValidationError if checks fail
    #     except csv.Error:
    #         raise ValidationError('Failed to parse the CSV file')


DATASET_CHOICES = (
    ("flowers_dataset", "flowers_dataset"),
    ("UNM_dataset", "UNM_dataset"),
    ("NEU_dataset", "NEU_dataset"),
    ("Dartmouth_dataset", "Dartmouth_dataset")
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
    dataset_file = CSVFileField(help_text='Only csv files are accepted.')
