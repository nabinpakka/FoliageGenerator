import os
import random
from PIL import Image
import numpy as np
from pathlib import Path


def get_random_images(folder_path, num_images=40):
    """
    Get random images from a folder

    Args:
        folder_path (str): Path to the folder containing images
        num_images (int): Number of images to select

    Returns:
        list: List of paths to randomly selected images
    """
    # Get all image files from the folder
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]

    # Select random images
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images found in folder")
        return [os.path.join(folder_path, f) for f in image_files]

    selected_files = random.sample(image_files, num_images)
    return [os.path.join(folder_path, f) for f in selected_files]


def crop_image_to_squares(image_path, output_dir, target_size=1024):
    """
    Crop an image into 4 equal squares of target_size x target_size

    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save cropped images
        target_size (int): Size of the square crops
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open and resize image
    img = Image.open(image_path)

    # Calculate the minimum dimension needed for 2x2 grid of target_size
    min_dimension = target_size * 2

    # Resize image maintaining aspect ratio so shortest side is min_dimension
    aspect_ratio = img.size[0] / img.size[1]
    if aspect_ratio > 1:
        new_size = (int(min_dimension * aspect_ratio), min_dimension)
    else:
        new_size = (min_dimension, int(min_dimension / aspect_ratio))

    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Calculate starting coordinates for center crop
    left = (img.size[0] - min_dimension) // 2
    top = (img.size[1] - min_dimension) // 2

    # Crop to center square
    img = img.crop((left, top, left + min_dimension, top + min_dimension))

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Crop into 4 equal squares
    crops = []
    for i in range(2):
        for j in range(2):
            # Calculate coordinates for each crop
            crop_left = j * target_size
            crop_top = i * target_size
            crop_right = crop_left + target_size
            crop_bottom = crop_top + target_size

            # Crop and save
            crop = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            output_path = os.path.join(output_dir, f"{base_name}_crop_{i}_{j}.png")
            crop.save(output_path)
            crops.append(output_path)

    return crops


def process_directory(input_dir, output_dir, num_images=40):
    """
    Process a single directory of images

    Args:
        input_dir (str): Path to input directory
        output_dir (str): Path to output directory
        num_images (int): Number of images to process
    """
    # Get random images
    image_paths = get_random_images(input_dir, num_images)

    # Process each image
    all_crops = []
    for img_path in image_paths:
        try:
            crops = crop_image_to_squares(img_path, output_dir)
            all_crops.extend(crops)
            print(f"Processed: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    return len(image_paths), len(all_crops)


def process_multiple_directories(base_input_path, base_output_path, num_images=40):
    """
    Process multiple directories of images while maintaining directory structure

    Args:
        base_input_path (str): Base path containing multiple directories
        base_output_path (str): Base path for output
        num_images (int): Number of images to process per directory
    """
    # Convert to Path objects for easier handling
    base_input = Path(base_input_path)
    base_output = Path(base_output_path)

    # Create base output directory if it doesn't exist
    base_output.mkdir(parents=True, exist_ok=True)

    total_dirs = 0
    total_images = 0
    total_crops = 0

    # Walk through all directories in the input path
    for dir_path, subdirs, files in os.walk(base_input):
        # Skip if no image files in directory
        if not any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) for f in files):
            continue

        # Create corresponding output directory path
        relative_path = Path(dir_path).relative_to(base_input)
        output_dir = base_output / relative_path

        print(f"\nProcessing directory: {dir_path}")
        print(f"Saving crops to: {output_dir}")

        # Process the directory
        images_processed, crops_created = process_directory(dir_path, output_dir, num_images)

        if images_processed > 0:
            total_dirs += 1
            total_images += images_processed
            total_crops += crops_created

    # Print summary
    print("\nProcessing Complete!")
    print(f"Processed {total_dirs} directories")
    print(f"Total images processed: {total_images}")
    print(f"Total crops created: {total_crops}")


# Example usage
if __name__ == "__main__":
    base_input_path = "/home/nabin/Documents/DiseaseClassification/src/natural_patch"
    base_output_path = "/home/nabin/Documents/DiseaseClassification/src/natural_foliage_1024"
    process_multiple_directories(base_input_path, base_output_path, 280)
