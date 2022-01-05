import requests


def upload_file(uploader_name, uploader_email, dataset_type, f):
    url = "http://api:8887/query/dataset-upload/"

    payload = {'uploader_name': uploader_name,
               'uploader_email': uploader_email,
               'dataset_type': dataset_type}

    files = [('dataset_file', f.open(mode='rb'))]
    headers = {}

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files)

    return response

def handle_csv_only_file(uploader_name, uploader_email, dataset_type, f):

    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response

def handle_flowers_file(uploader_name, uploader_email, dataset_type, f):
    #default_storage.save('datasets/flowers/flowers.csv', f)

    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response


def handle_unm_file(uploader_name, uploader_email, dataset_type, f):
    #default_storage.save('datasets/unm/unm.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response


def handle_neu_file(uploader_name, uploader_email, dataset_type, f):
    # default_storage.save('datasets/neu/neu.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response


def handle_dartmouth_file(uploader_name, uploader_email, dataset_type, f):
    # default_storage.save('datasets/dartmouth/dartmouth.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response

##JAG how do I change default storage.save ??
def handle_nhanes_bio_file(uploader_name, uploader_email, dataset_type, f):
    # default_storage.save('datasets/dartmouth/dartmouth.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response

def handle_nhanes_llod_file(uploader_name, uploader_email, dataset_type, f):
    # default_storage.save('datasets/dartmouth/dartmouth.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response

def handle_nhanes_dd_file(uploader_name, uploader_email, dataset_type, f):
    # default_storage.save('datasets/dartmouth/dartmouth.csv', f)

    # TODO - Validate csv header and then upload

    # Upload to api
    response = upload_file(uploader_name, uploader_email, dataset_type, f)
    return response