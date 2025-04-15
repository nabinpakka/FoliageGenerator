import os
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
from argparse import Namespace

from PIL import Image

from src.model.foliage import Foliage
from src.utilities import Utility


def create_patch_images_for_disease(base_output_path, disease="frogeye", thread_num=2):
    num_patch_images_per_disease = 200
    for i in range(0, num_patch_images_per_disease):
        patch_image: Image = foliage.get_patch_of_leaves(disease)

        disease_dir = os.path.join(base_output_path, disease)

        output_path = disease_dir + "/" + str(thread_num) + "_" + str(i) + ".png"
        print("Saved image at: ", output_path)
        patch_image.save(output_path)


def create_dir_if_not_exist(path, disease_list):
    for disease in disease_list:
        disease_dir = os.path.join(path, disease)
        os.makedirs(disease_dir, exist_ok=True)

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', required=True, help="Plant specific configuration file. A json file")
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    utility = Utility()
    args = parse_arguments()

    config = utility.json_parser(args.config)
    diseases = utility.string_to_list(config.get("diseases"))

    base_output_path = config.get("output_path")

    foliage = Foliage(config)

    create_dir_if_not_exist(base_output_path, diseases)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(create_patch_images_for_disease, base_output_path, disease, idx) for idx, disease in enumerate(diseases)]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time: 0.2f}")

