from django.views.generic import TemplateView, FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseRedirect

from .forms import UploadFileForm

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def image_upload(request):
    if request.method == "POST" and request.FILES["image_file"]:
        image_file = request.FILES["image_file"]
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_url = fs.url(filename)
        print(image_url)
        return render(request, "upload.html", {
            "image_url": image_url
        })
    return render(request, "upload.html")


def handle_uploaded_file(f):
    print("This is running!")
    with open('upload-name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


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
        print('>>>> Inside Upload File')
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                handle_uploaded_file(request.FILES['dataset_file'])
                return HttpResponseRedirect('/upload/success/')
        else:
            form = UploadFileForm()
        return HttpResponse(request, 'upload.html', {'form': form})
