from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage

from .forms import UploadFileForm
from .tools.uploads import *


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
        response = HttpResponse()
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                uploader_name = request.POST.get('uploader_name')
                uploader_email = request.POST.get('uploader_email')
                dataset_type = request.POST.get('dataset_type')
                f = request.FILES['dataset_file']

                if (dataset_type == 'flowers_dataset'):
                    # print('Got Flowers Dataset')
                    response = handle_flowers_file(
                        uploader_name, uploader_email, dataset_type, f)

                if (dataset_type == 'UNM_dataset'):
                    # print('Got UNM Dataset')
                    respose = handle_unm_file(f)

                if (dataset_type == 'NEU_dataset'):
                    # print('Got NEU Dataset')
                    response = handle_neu_file(f)

                if (dataset_type == 'Dartmouth_dataset'):
                    # print('Got Dartmouth Dataset')
                    response = handle_dartmouth_file(f)

                if response.status_code == 201:
                    return HttpResponseRedirect('/upload/success/')
                else:
                    return response
        else:
            form = UploadFileForm()
        return HttpResponse(request, 'upload.html', {'form': form})
