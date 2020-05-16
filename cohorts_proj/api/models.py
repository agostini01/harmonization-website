from django.db import models

DATASET_CHOICES = (
    ("flowers_dataset", "flowers_dataset"),
    ("UNM_dataset", "UNM_dataset"),
    ("NEU_dataset", "NEU_dataset"),
    ("Dartmouth_dataset", "Dartmouth_dataset")
)


class DatasetUploadModel(models.Model):
    uploader_name = models.CharField(max_length=100)
    uploader_email = models.EmailField(max_length=100)
    dataset_time = models.DateTimeField(auto_now=True)
    dataset_type = models.CharField(max_length=100, choices=DATASET_CHOICES)
    dataset_file = models.FileField(upload_to='raw-datasets/%Y/%m/%d/')

    def __str__(self):
        return self.title
