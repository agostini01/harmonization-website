from django import forms

DATASET_CHOICES = (
    ("flowers_dataset", "flowers_dataset"),
    ("UNM_dataset", "UNM_dataset"),
    ("NEU_dataset", "NEU_dataset"),
    ("Dartmouth_dataset", "Dartmouth_dataset")
)


class UploadFileForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)

    uploader_name = forms.CharField(max_length=100,
                                    help_text="Scientist uploading this dataset.")
    uploader_email = forms.EmailField(
        help_text='A valid email address, please.')
    dataset_type = forms.ChoiceField(choices=DATASET_CHOICES)
    dataset_file = forms.FileField(allow_empty_file=False)
