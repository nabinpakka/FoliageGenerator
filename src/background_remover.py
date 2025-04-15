import os
from typing import List

import requests
from PIL import Image
from rembg import remove

API_KEY = 'BV3EcxHsd123JNkWjicALzCx'


def request_remove_background_bgai(image_path, output_path):
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



def request_remove_background_picsart(image_path, output_path):
    import requests

    url = "https://api.picsart.io/tools/1.0/removebg"

    payload = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"output_type\"\r\n\r\ncutout\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"bg_blur\"\r\n\r\n0\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"scale\"\r\n\r\nfit\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"auto_center\"\r\n\r\nfalse\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"stroke_size\"\r\n\r\n0\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"stroke_color\"\r\n\r\nFFFFFF\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"stroke_opacity\"\r\n\r\n100\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"shadow\"\r\n\r\ndisabled\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"shadow_opacity\"\r\n\r\n20\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"shadow_blur\"\r\n\r\n50\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"format\"\r\n\r\nPNG\r\n-----011000010111000001101001--"
    headers = {
        "accept": "application/json",
        "content-type": "multipart/form-data; boundary=---011000010111000001101001"
    }

    response = requests.post(url, data=payload, headers=headers)

    print(response.text)

def remove_background_rembg(image_path, output_path):
    input = Image.open(image_path)
    output = remove(input)
    output = output.convert('RGBA')
    print("File saved: ", output_path)
    output.save(output_path)

def remove_background_dir(dir_path, output_path, dir):

    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        output_dir_path = os.path.join(output_path, dir)
        output_file_path = os.path.join(output_dir_path, file)

        #convert jpg to png
        output_file_path = output_file_path.replace(".JPG",".png")

        # create dir if not exist
        os.makedirs(output_dir_path, exist_ok=True)

        remove_background_rembg(file_path, output_file_path)


def remove_background(input_path, output_path):
    # listing directories
    for dir in os.listdir(input_path):
        dir_path = os.path.join(input_path, dir)
        remove_background_dir(dir_path, output_path, dir)


if __name__ == '__main__':
    base_image_path = "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/images/tomato"
    base_output_path = "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/output_images/tomato"

    base_output_path.replace()

    remove_background(base_image_path, base_output_path)