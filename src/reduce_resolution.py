import os
import random

from PIL import Image

def crop_image(base_image_path, base_output_path, crop_size=(1024,1024), img_size = (512,512)):
    # completed = ["rust", "cercospora_leaf_blight", "downey_mildew", "mosiac_virus", "target_spot", "healthy"]
    completed =["bacterial_blight", "healthy", "frogeye", "sudden_death_syndrom", "rust", "mosiac_virus", "downey_mildew", "cercospora_leaf_blight", "potassium_deficiency"]
    for dir in os.listdir(base_image_path):
        # if dir in completed:
        #     continue
        

        # take only first 200 files
        file_count = 0

        os.makedirs(os.path.join(base_output_path, dir), exist_ok = True)
        # introduce randomness while collecting first X files
        file_list = os.listdir(os.path.join(base_dir, dir))
        random.shuffle(file_list)
        for filename in file_list:
            image_path = os.path.join(base_image_path, dir, filename)
            with Image.open(image_path) as img:
                # Calculate cropping box
                width, height = img.size
                crop_width, crop_height = crop_size
                for i in range (0, 2):
                    left = (width - crop_width) // 2 
                    top = 0 + i * 400
                    right = left + crop_width
                    bottom = top + crop_height

                    cropped_image = img.crop((left, top, right, bottom))
                    # resized_img = cropped_image.resize(img_size, Image.LANCZOS)

                    split_name = filename.split(".")
                    filename_new = split_name[0] + "_0" + str(i) + "." + split_name[1] 
                    output_path = os.path.join(base_output_path, dir, filename_new)
                    print(output_path)
                    cropped_image.save(output_path)
                file_count +=1
                if file_count > 400:
                    break



def resize_images_in_directory(input_dir, output_dir, size=(512, 512)):
    """
    Resize all images in the input directory to the specified size and save them in the output directory.
    
    :param input_dir: Path to the directory containing input images.
    :param output_dir: Path to the directory where resized images will be saved.
    :param size: Tuple specifying the desired resolution (width, height).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Check if the file is an image
        try:
            with Image.open(input_path) as img:
                # Resize the image
                resized_img = img.resize(size, Image.LANCZOS)
                # Save the resized image
                resized_img.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Skipping file {filename}: {e}")

if __name__ == '__main__':
    base_dir = "path_to_input_dir"
    output_dir = "path_to_output_dir"


    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     futures = [executor.submit(create_patch_images_for_disease, disease, i) for i in range(5,10)]

    crop_image(base_dir, output_dir)

