import os
from typing import List

import requests

API_KEY = 'BV3EcxHsd123JNkWjicALzCx'


def request_remove_background(image_path, output_path):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(image_path, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': API_KEY},
    )
    if response.status_code == requests.codes.ok:
        with open(output_path, 'wb') as out:
            out.write(response.content)
            print("Removed background of ", image_path)
    else:
        print("Error:", response.status_code, response.text)

def remove_background_dir(dir_path, output_path, dir):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        output_dir_path = os.path.join(output_path, dir)
        output_file_path = os.path.join(output_dir_path, file)

        # create dir if not exist
        os.makedirs(output_dir_path, exist_ok=True)

        request_remove_background(file_path, output_file_path)


def remove_background(input_path, output_path):
    # listing directories
    for dir in os.listdir(input_path):
        dir_path = os.path.join(input_path, dir)
        remove_background_dir(dir_path, output_path, dir)


if __name__ == '__main__':
    base_image_path = "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/images/tomato"
    base_output_path = "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/output_images/tomato"

    remove_background(base_image_path, base_output_path)