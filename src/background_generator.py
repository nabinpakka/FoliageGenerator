import os
from PIL import Image


def crop_image_with_sliding_window(input_image_path, crop_width=750, crop_height=750,
                                   slide_x_step=100, slide_y_step=100,
                                   output_dir='cropped_images', crop_count = 0):
    """
    Crop a large image into multiple smaller images using a sliding window approach.

    :param input_image_path: Path to the source image
    :param crop_width: Width of the cropped images (default 500)
    :param crop_height: Height of the cropped images (default 500)
    :param slide_x_step: Horizontal sliding step (default 100)
    :param slide_y_step: Vertical sliding step (default 100)
    :param output_dir: Directory to save cropped images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the input image
    with Image.open(input_image_path) as img:
        # Get image dimensions
        img_width, img_height = img.size

        # Slide horizontally first
        for y in range(0, img_height - crop_height + 1, slide_y_step):
            for x in range(0, img_width - crop_width + 1, slide_x_step):
                # Crop the image
                crop = img.crop((x, y, x + crop_width, y + crop_height))

                crop_count += 1
                # Generate filename
                output_filename = os.path.join(
                    output_dir,
                    f'crop_{crop_count:04d}.png'
                )

                # Save the cropped image
                crop.save(output_filename)
                print(f"Saved {output_filename}")

        print(f"Total crops created: {crop_count}")
        return crop_count


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    input_image_path = "path_to_background_image"

    # Crop the image
    crop_count = 0
    crop_count = crop_image_with_sliding_window(
        input_image_path,
        crop_width=1500,
        crop_height=1500,
        slide_x_step=300,
        slide_y_step=300,
        crop_count=crop_count
    )