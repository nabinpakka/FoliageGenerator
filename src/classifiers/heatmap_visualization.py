# '''Attention Visualization'''
# """First, test the pre-trained network"""
import paddle
import paddle.vision.transforms as T
import cv2
from Grad_CAM import Display
from matplotlib import pyplot as plt
import os

from conv_next import ConvNeXt

paddle.set_device("GPU:1")

model = ConvNeXt(num_classes=10, num_patch=8)
print(model)
model_state_dict = paddle.load(
    'path_to_model')
model.set_state_dict(model_state_dict)

# Display.show_network(model)

# 1. Specify the convolution layer
# """Note: You can directly specify the convolution layer"""
layer = model.attention_layers[0]

# 2. Instantiate a Display object
display = Display(model, layer)


def find_first_image_path(base_dir):
    first_images = []
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)

        # Ensure it's a directory
        if os.path.isdir(class_path):
            # Find image files (you can extend this list)
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']

            # Find first image in the directory
            for file in os.listdir(class_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    first_image_path = os.path.join(class_path, file)
                    first_images.append((class_dir, first_image_path))
                    break  # Stop after finding first image

    return first_images


# img_folder = 'work/infer'
# img_list = os.listdir(img_folder)
# img_list = filter(lambda x: '.png' in x, img_list)
base_dir = "path_to_data"
img_paths = find_first_image_path(base_dir)

for class_name, img_path in img_paths:
    img = cv2.imread(img_path)
    img = img[:, :, ::-1]  # Convert BGR to RGB
    save_dir = os.path.join('./results/', 'heatmap/', 'best')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'heatmap_{class_name}')
    # 3. Save the model attention map
    display.save(img, file=save_path)


