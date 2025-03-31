import os
from concurrent.futures import ThreadPoolExecutor
import time

from PIL import Image

from trifoliate_patch import TrifoliatePatch

BASE_OUTPUT_PATH = "/home/nabin/Documents/DiseaseClassification/src/output_images/new_image95"


patch = TrifoliatePatch()
def create_patch_images_for_disease(disease="frogeye", thread_num = 2):

    num_patch_images_per_disease = 200
    for i in range(0, num_patch_images_per_disease):
        patch_image: Image = patch.get_patch_of_trifoliate(disease)

        disease_dir = os.path.join(BASE_OUTPUT_PATH , disease)

        output_path = disease_dir +"/"+ str(thread_num) + "_" + str(i) +".png"
        print("Saved image at: ", output_path)
        patch_image.save(output_path)

def create_dir_if_not_exist(disease_list):
    for disease in disease_list:
        disease_dir = os.path.join(BASE_OUTPUT_PATH , disease)
        os.makedirs(disease_dir, exist_ok=True)
            

if __name__ == '__main__':

    start_time = time.time()
    # diseases = ["frogeye", "bacterial_blight", "rust", "cercospora_leaf_blight", "downey_mildew", "mosiac_virus", "potassium_deficiency", "sudden_death_syndrom", "target_spot", "healthy"]

    diseases = [ "mosiac_virus", "potassium_deficiency", "sudden_death_syndrom", "target_spot", "healthy","mosiac_virus", "potassium_deficiency", "sudden_death_syndrom", "target_spot", "healthy","mosiac_virus", "potassium_deficiency", "sudden_death_syndrom", "target_spot", "healthy" ]

    create_dir_if_not_exist(diseases)
    # disease = "potassium_deficiency"
    # multi threading
    # create_patch_images_for_disease(disease, 12)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(create_patch_images_for_disease, disease, idx) for idx, disease in enumerate(diseases)]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time: 0.2f}")

 