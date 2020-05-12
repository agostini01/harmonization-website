from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect

from .forms import UploadFileForm

from django.core.files.storage import default_storage


def handle_flowers_file(f):
    # TODO
    default_storage.save('datasets/flowers/flowers.csv', f)


def handle_unm_file(f):
    # TODO
    default_storage.save('datasets/unm/unm.csv', f)


def handle_neu_file(f):
    # TODO
    default_storage.save('datasets/neu/neu.csv', f)


def handle_dartmouth_file(f):
    # TODO
    default_storage.save('datasets/dartmouth/dartmouth.csv', f)


class UploadSuccessPageView(LoginRequiredMixin, TemplateView):
    template_name = 'upload-success.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'


class UploadPageView(LoginRequiredMixin, FormView):
    template_name = 'upload.html'
    login_url = '/accounts/login/'
    redirect_field_name = 'redirect'
    success_url = '/upload/success/'

    form_class = UploadFileForm

    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                dataset_type = request.POST.get('dataset_type')
                f = request.FILES['dataset_file']

                if (dataset_type == 'flowers_dataset'):
                    print('Got Flowers Dataset')
                    handle_flowers_file(f)

                if (dataset_type == 'UNM_dataset'):
                    print('Got UNM Dataset')
                    handle_unm_file(f)

                if (dataset_type == 'NEU_dataset'):
                    print('Got NEU Dataset')
                    handle_neu_file(f)

                if (dataset_type == 'Dartmouth_dataset'):
                    print('Got Dartmouth Dataset')
                    handle_dartmouth_file(f)

                return HttpResponseRedirect('/upload/success/')
        else:
            form = UploadFileForm()
        return HttpResponse(request, 'upload.html', {'form': form})
