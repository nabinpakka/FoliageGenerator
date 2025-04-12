import os
import argparse
from PIL import Image

def convert_images_to_pdfs(input_dir, output_dir, extensions=('jpg', 'jpeg', 'png')):
    """
    Convert all images in a directory to individual PDF files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all files in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.pdf")

            try:
                with Image.open(input_path) as img:
                    # Convert image to RGB mode if needed (for CMYK images)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # reshaping the image
                    img = img.resize((1024,1024))
                    # Save as PDF
                    img.save(output_path, "PDF", resolution=100.0)
                    print(f"Converted: {filename} -> {os.path.basename(output_path)}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == '__main__':
    input_dir = "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/src/pdf_images"
    output_dir ="/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/src/pdf_images"
    convert_images_to_pdfs(input_dir, output_dir)