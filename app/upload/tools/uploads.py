import requests


def upload_file(uploader_name, uploader_email, dataset_type, f):
    url = "http://api:8888/query/dataset-upload/"

    payload = {'uploader_name': uploader_name,
               'uploader_email': uploader_email,
               'dataset_type': dataset_type}

    files = [('dataset_file', f.open(mode='rb'))]
    headers = {}

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files)

    return response


def handle_flowers_file(uploader_name, uploader_email, dataset_type, f):
    #default_storage.save('datasets/flowers/flowers.csv', f)

    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response


def handle_unm_file(f):
    # TODO
    default_storage.save('datasets/unm/unm.csv', f)


def handle_neu_file(f):
    # TODO
    default_storage.save('datasets/neu/neu.csv', f)


def handle_dartmouth_file(f):
    # TODO
    default_storage.save('datasets/dartmouth/dartmouth.csv', f)
