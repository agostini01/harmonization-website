from django.core.exceptions import ValidationError

# import csv


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
