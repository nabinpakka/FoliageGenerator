import math

from PIL import Image
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
        self.PLANT_IMAGE_SIZE = (
            config.get("single_plant_size"), config.get("single_plant_size"))  # y-axis height of plant in the image
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

    def get_single_plant(self, disease="frogeye"):
        num_leaves = random.randint(self.LEAF_RANGE[0], self.LEAF_RANGE[1])

        # background transparent image
        background = Image.new("RGBA", self.PLANT_IMAGE_SIZE, (0, 0, 0, 0))

        stem_coords = (self.PLANT_IMAGE_SIZE[0] // 2, self.PLANT_IMAGE_SIZE[1] // 2)
        center = (
        self.PLANT_IMAGE_SIZE[0] // 2 - self.CENTER_OFFSET, self.PLANT_IMAGE_SIZE[1] // 2 - self.CENTER_OFFSET)
        coords = self.plant.get_leaf_arrangement_coords(center, num_leaves)
        angles = self.plant.get_angle_of_rotation_for_coords(center, coords)

        # percentage of healthy leaves in a single plant
        rand_healthy_leaves_percent = 100 - self.DISEASE_PERCENT
        healthy_leaves_count = math.ceil(num_leaves * rand_healthy_leaves_percent / 100)

        # trifoliate to cover the gap in the center of the single plant
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
            background.paste(leaves, coord, leaves)
        return background


if __name__ == '__main__':
    utility = Utility()
    config = utility.json_parser(
        "/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/src/config.json")
    single_plant = SinglePlant( config)

    start_time = time.time()
    single_plant.get_single_plant().show()
    end_time = time.time()
    print("Time for single plant: ", end_time - start_time)
