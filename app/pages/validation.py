from django.http import HttpResponse

from .forms import PLOT_TYPES
from .choices.har import ADDITIONAL_FEATURES

VALIDATE_ERRORS = (
    (0, 'Success'),
    (1, 'Categorical Plot Limitation')
)


def checkFormRequest(request):
    """Check if the requested form has an error."""

    plot_type = request.GET.get('plot_type')
    x_feature = request.GET.get('x_feature')

    if plot_type in [x[0] for x in PLOT_TYPES[1][1]]:

        if x_feature in [x[0] for x in ADDITIONAL_FEATURES[0][1]]:
            pass

        else:
            return(VALIDATE_ERRORS[1])

    return(VALIDATE_ERRORS[0])


def getErrorImage(error):
    """Parse error and create a HttpResponse with correct image data.

    error - tuple of (integer error value, string)"""
    path = ''

    if(error[0] == 1):
        print("Recovered ERROR: Tried to select a non categorical X feature "
              "with a categorical plot.")
        path = '/usr/src/app/static/images/errors/cat-mismatch.png'

    with open(path, 'rb') as f:
        image_data = f.read()
    return HttpResponse(image_data, content_type="image/png")
