import math

from PIL import Image
import random
import time

from src.soybean.trifoliate import Trifoliate
from src.utilities import Utility


class SinglePlant:
    def __init__(self, config):

        num_leaves = config.get("num_leaves")
        self.LEAF_RANGE = (num_leaves - 5, num_leaves + 5)
        self.PLANT_IMAGE_SIZE = (config.get("single_plant_size"), config.get("single_plant_size"))  # y-axis height of plant in the image
        self.PLANT_IMAGE_RADIUS = 30
        self.CENTER_OFFSET = 100
        self.DISEASE_PERCENT = config.get("disease_rate")
        # self.leaf = leaf
        self._setup()

    def _setup(self):
        self.utility = Utility()
        self.trifoliate = Trifoliate("dd")


    def get_angle_of_rotation_for_coords(self, center, coords):
        angles = []
        x2, y2 = center
        for coord in coords:
            x1, y1 = coord

            dx = x2 - x1
            dy = y2 - y1

            if dy==0:
                angle = 90 if dx > 0 else -90
                dy = 0.00000001
            angle_rad = math.atan(dx / dy)
            angle_deg = math.degrees(angle_rad)

            # adjusting angle based on quadrant
            if dy < 0:
                angle_deg += 180 if dx < 0 else -180
            angle = angle_deg
            angles.append(angle)
        return angles

    def get_single_plant(self, disease="frogeye"):
        num_trifoliates = random.randint(self.LEAF_RANGE[0], self.LEAF_RANGE[1])

        # background transparent image
        background = Image.new("RGBA", self.PLANT_IMAGE_SIZE, (0, 0, 0, 0))

        # get random angle
        bifoliate_angle = self.utility.get_random_angle(1)
        # angles = self.utility.get_random_angle(num_trifoliates, 360//num_trifoliates)
        stem_coords = (self.PLANT_IMAGE_SIZE[0]//2 , self.PLANT_IMAGE_SIZE[1]//2 )
        center = (self.PLANT_IMAGE_SIZE[0]//2 - self.CENTER_OFFSET, self.PLANT_IMAGE_SIZE[1]//2 - self.CENTER_OFFSET)
        coords = self.utility.get_phyllotactic_leaf_arrangement_coords(center,  num_trifoliates)
        angles = self.get_angle_of_rotation_for_coords(center, coords)

        # percentage of healthy trifoliate as most of the leaves will be healthy
        rand_healthy_trifoliate_percent = 100 - self.DISEASE_PERCENT
        healthy_trifoliate_count = math.ceil(num_trifoliates * rand_healthy_trifoliate_percent / 100)

        # trifoliate to cover the gap in the center of the single plant
        center_trifoliate = self.trifoliate.get_leaves(disease, angles[0])
        background.paste(center_trifoliate, center, center_trifoliate)

        num_trifoliates_to_scale = 5
        scale_factor = 0.5
        scale_offset = 0.05

        # Second layer
        # Second layer contains trifoliates on the main stem only
        # The number of trifoliates vary from 2-4
        # num_trifoliates_no_branch = random.randint(2,4)
        for idx, (coord, angle) in enumerate(zip(coords, angles)):
            disease_type = "healthy" if idx % 2 == 0 and healthy_trifoliate_count > 0 else disease
            if disease_type == "healthy":
                healthy_trifoliate_count -= 1
            if num_trifoliates_to_scale > 0:
                scale_factor = scale_factor + scale_offset * idx
                trifoliate = self.trifoliate.get_leaves(disease_type, angle, scale_factor)
                num_trifoliates_to_scale -= 1
            else:
                trifoliate = self.trifoliate.get_leaves(disease_type, angle)
            background.paste(trifoliate, coord, trifoliate)
            # self._draw_branch(background, stem_coords, coord)
        return background

if __name__ == '__main__':
    single_plant = SinglePlant()

    start_time = time.time()
    single_plant.get_single_plant().show()
    end_time = time.time()
    print("Time for single plant: ", end_time - start_time)

