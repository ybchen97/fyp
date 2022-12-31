import os
import requests

def download_image(img_url, save_dir):
    try:        
        # check if exists
        file = os.path.join(save_dir, os.path.basename(img_url))
        if os.path.exists(file): # exists
            return

        # get file
        img_bytes = requests.get(img_url).content
        file_name = os.path.basename(img_url)

        # name
        file_path = os.path.join(save_dir, file_name)

        # save
        with open(file_path, 'wb') as img_file:
            img_file.write(img_bytes)
            print(f'\r{file_name} was downloaded...', end = '', flush = True)
    except Exception as e:
        print(e)
