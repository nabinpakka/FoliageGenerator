import os
from typing import List

import requests
from PIL import Image
from rembg import remove


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

        # convert jpg to png
        output_file_path = output_file_path.replace(".JPG", ".png")

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
