import concurrent.futures
import os

import firebase_admin
import requests
from firebase_admin import storage, firestore
from firebase_admin.firestore import client


def is_firebase_initialized():
    try:
        # Try to get the default app
        firebase_admin.get_app()
        # If this point is reached, firebase has been initialized.
        return True
    except ValueError as e:
        # If a ValueError is raised, firebase has not been initialized.
        return False


def upload_file_to_firebase(file_path, destination_path):
    """
    Uploads a file to Firebase Storage and returns the downloadable URL.

    :param file_path: str, Local path to the file to be uploaded
    :param destination_path: str, Path in Firebase Storage where the file should be uploaded
    :return: str, Downloadable URL
    """
    bucket = storage.bucket()
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file_path)

    # Make the file publicly accessible
    blob.make_public()

    # Get the downloadable URL
    download_url = blob.public_url

    return download_url


# def download_file_from_url(url, local_path):
#     """
#     Downloads a file from the given URL and saves it to the specified local path.
#
#     :param url: str, The URL of the file to download
#     :param local_path: str, The local path where the file should be saved
#     """
#
#     # Send a HTTP request to the URL of the file, stream=True ensures that the file is downloaded efficiently
#     encoded_url = quote(url, safe=':/?&=')
#     response = requests.get(encoded_url, stream=True)
#
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Open a binary file in write mode
#         with open(local_path, 'wb') as file:
#             # Write the contents of the response to the file
#             for chunk in response.iter_content(chunk_size=1024):
#                 file.write(chunk)
#         print('File downloaded successfully')
#     else:
#         print(f'Failed to download the file from {url}, status code {response.status_code}, {response.reason}.')

def download_file_from_url(url, local_path):
    """
    Downloads a file from the given URL and saves it to the specified local path.

    :param url: str, The URL of the file to download
    :param local_path: str, The local path where the file should be saved
    """
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        # Open a binary file in write mode
        with open(local_path, 'wb') as file:
            # Write the contents of the response to the file
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print('File downloaded successfully')
    except requests.exceptions.RequestException as e:
        print(f'Failed to download the file from {url}. Error: {e}')


def upload_folder_to_storage(local_folder_path, bucket_folder_path):
    urls = {}
    for root, _, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            storage_path = os.path.join(bucket_folder_path, relative_path)
            urls[file.split('.')[0]] = upload_file_to_firebase(local_file_path, storage_path)
    return urls


def update_docs(url, dance_id, user_id):
    db = firestore.client()
    dance_ref = db.collection('users').document(user_id).collection('dances').document(dance_id)

    # Use set with merge=True to ensure the document is created or updated
    dance_ref.set({
        'status': 'completed',
        'video_url': url,
        'updated_at': firestore.SERVER_TIMESTAMP
    }, merge=True)

    return url


def update_docs_for_preprocessing(url, dance_id, user_id):
    db = firestore.client()
    dance_ref = db.collection('users').document(user_id).collection('dances').document(dance_id)

    # Use set with merge=True to ensure the document is created or updated
    dance_ref.set({
        'status': 'completed',
        'preprocessing_url': url,
        'updated_at': firestore.SERVER_TIMESTAMP
    }, merge=True)

    return url


def download_folder_from_storage(bucket_folder_path, local_folder_path):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=bucket_folder_path)
    for blob in blobs:
        # Skip directories
        if not blob.name.endswith('/'):
            relative_path = os.path.relpath(blob.name, bucket_folder_path)
            local_file_path = os.path.join(local_folder_path, relative_path)

            # Create local directory structure if it doesn't exist
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download the file
            blob.download_to_filename(local_file_path)
            print(f'Downloaded {blob.name} to {local_file_path}')


def download_blob(blob, local_folder_path, bucket_folder_path):
    if not blob.name.endswith('/'):
        relative_path = os.path.relpath(blob.name, bucket_folder_path)
        local_file_path = os.path.join(local_folder_path, relative_path)

        # Create local directory structure if it doesn't exist
        local_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the file
        blob.download_to_filename(local_file_path)
        print(f'Downloaded {blob.name} to {local_file_path}')


def download_folder_from_storage_in_parallel(bucket_folder_path, local_folder_path):
    bucket = storage.bucket()  # Replace with your bucket name
    blobs = bucket.list_blobs(prefix=bucket_folder_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_blob, blob, local_folder_path, bucket_folder_path) for blob in blobs]
        concurrent.futures.wait(futures)


def update_docs_for_failure(user_id, dance_id):
    db = client()
    dance_ref = db.collection('users').document(user_id).collection('dances').document(dance_id)
    dance_ref.set({
        'status': 'failed',
        'video_url': '',
        'updated_at': firestore.SERVER_TIMESTAMP
    }, merge=True)
    return ''
