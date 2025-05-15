import math

from PIL import Image, ImageDraw
import random
import time

from src.soybean.soybean import Soybean
from src.tomato.tomato import Tomato
from src.utilities import Utility


class SinglePlant:
    def __init__(self, config):

        num_leaves = config.get("num_leaves")
        self.plant_type: str = config.get("type")
        self.LEAF_RANGE = (num_leaves - 5, num_leaves + 5)
        self.PLANT_IMAGE_SIZE = (config.get("single_plant_size"), config.get("single_plant_size"))  # y-axis height of plant in the image
        self.PLANT_IMAGE_RADIUS = 30
        self.CENTER_OFFSET = 100
        self.DISEASE_PERCENT = config.get("disease_rate")
        # self.leaf = leaf
        self._setup(config)

    def _setup(self, config):
        self.utility = Utility()
        self.__get_plant_specific_object(config)

    def __get_plant_specific_object(self, config):
        if self.plant_type.lower() == "soybean":
            self.plant = Soybean(config)
        elif self.plant_type.lower() == "tomato":
            self.plant = Tomato(config)

    def place_rotated_image(self, img, target_point,
                            rotation_angle, background):

        # Get the dimensions of the original image
        orig_width, orig_height = img.size

        # Rotate the image
        rotated_image = img.rotate(rotation_angle, expand=False, resample=Image.Resampling.BICUBIC)
        rotated_width, rotated_height = rotated_image.size

        target_x, target_y = target_point


        rotation_angle = (rotation_angle + 360) % 360

        # Determine which quadrant the angle falls into
        if 0 <= rotation_angle <= 90:
            # First quadrant: 0° to 90° -> 126 to 256 (increasing)
            min_angle, max_angle = 0, 90
            min_value, max_value = 126, 256
            offset = min_value + (max_value - min_value) * (rotation_angle - min_angle) / (max_angle - min_angle)
            paste_x = int(target_x - offset +50)
            paste_y = int(offset - min_value)

        elif 90 < rotation_angle <= 180:
            # Second quadrant: 90° to 180° -> 256 to 126 (decreasing)
            min_angle, max_angle = 90, 180
            min_value, max_value = 256, 126  # Note: min_value > max_value for decreasing
            offset = min_value + (max_value - min_value) * (rotation_angle - min_angle) / (max_angle - min_angle)
            paste_x = int(target_x - min_value)
            paste_y = int(target_y - offset + max_value)

        elif 180 < rotation_angle <= 270:
            # Third quadrant: 180° to 270° -> 126 to 256 (increasing)
            min_angle, max_angle = 180, 270
            min_value, max_value = 126, 256
            offset =  min_value + (max_value - min_value) * (rotation_angle - min_angle) / (max_angle - min_angle)
            paste_x = int(offset)
            paste_y = int(target_y )

        else:  # 270 < normalized_angle < 360
            # Fourth quadrant: 270° to 360° -> 256 to 126 (decreasing)
            min_angle, max_angle = 270, 360
            min_value, max_value = 256, 126  # Note: min_value > max_value for decreasing
            offset = min_value + (max_value - min_value) * (rotation_angle - min_angle) / (max_angle - min_angle)
            paste_x = int(offset + 50)
            paste_y = int(offset - max_value)

        # Paste the rotated image onto the canvas at the calculated position
        background.paste(rotated_image, (paste_x, paste_y), rotated_image)

        return background
    def get_single_plant(self, disease="healthy"):
        num_leaves = random.randint(self.LEAF_RANGE[0], self.LEAF_RANGE[1])

        # background transparent image
        background = Image.new("RGBA", self.PLANT_IMAGE_SIZE, (0, 0, 0, 0))

        center = (self.PLANT_IMAGE_SIZE[0] // 2 - self.CENTER_OFFSET, self.PLANT_IMAGE_SIZE[1] // 2 - self.CENTER_OFFSET)
        coords = self.plant.get_leaf_arrangement_coords(center, num_leaves)
        angles = self.plant.get_angle_of_rotation_for_coords(center, coords)

        # percentage of healthy leaves in a single plant
        rand_healthy_leaves_percent = 100 - self.DISEASE_PERCENT
        healthy_leaves_count = math.ceil(num_leaves * rand_healthy_leaves_percent / 100)

        # leaf to cover the gap in the center of the single plant
        if self.plant_type == "soybean":
            central_leaf = self.plant.get_leaves(disease, angles[0])
            background.paste(central_leaf, center, central_leaf)

        num_leaves_to_scale = 5
        scale_factor = 0.5
        scale_offset = 0.05

        # Second layer
        for idx, (coord, angle) in enumerate(zip(coords, angles)):
            disease_type = "healthy" if idx % 2 == 0 and healthy_leaves_count > 0 else disease
            if disease_type == "healthy":
                healthy_leaves_count -= 1
            if num_leaves_to_scale > 0:
                scale_factor = scale_factor + scale_offset * idx
                leaves = self.plant.get_leaves(disease_type, angle, scale_factor)
                num_leaves_to_scale -= 1
            else:
                leaves = self.plant.get_leaves(disease_type, angle)
            if self.plant_type == "tomato":
                # original center
                center_tomato = (center[0] + self.CENTER_OFFSET, center[1] + self.CENTER_OFFSET)
                background = self.place_rotated_image(leaves, center_tomato, angle, background)

            else:
                background.paste(leaves, coord, leaves)

        return background


if __name__ == '__main__':
    utility = Utility()
    config = utility.json_parser(
        "path_to_config")
    single_plant = SinglePlant(config)

    start_time = time.time()
    single_plant.get_single_plant().show()
    end_time = time.time()
    print("Time for single plant: ", end_time - start_time)
