import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from PIL import Image
from typing import Dict

from utilities import Utility
from single_plant import SinglePlant
class TrifoliatePatch:
    def __init__(self):

        self.PLANT_PER_PATCH = 12
        self.BASE_IMAGE_SIZE = (1024, 1500)
        self.SINGLE_IMAGE_SIZE = (600, 600)
        self.OFFSET = 100
        self._setup()

    def _setup(self):
        self.BASE_BACKGROUND_IMAGE = "/home/nabin/Documents/DiseaseClassification/src/background_images/cropped_images"

        self._image_cache: Dict[str, Image.Image] = {}
        self.single_plant = SinglePlant()
        self.utility = Utility()
        self.background_paths = self.__get_list_of_background_images()
    def __get_list_of_background_images(self):
        """Pre-load all background image paths"""
        return [
            os.path.join(self.BASE_BACKGROUND_IMAGE, img)
            for img in os.listdir(self.BASE_BACKGROUND_IMAGE)
            if img.endswith('.png')
        ]

    # TODO unify the cache from multiple classes
    # create an image utility class to perform all image related common tasks from single class
    @lru_cache(maxsize=128)
    def _load_and_prepare_image(self, image_path: str) -> Image.Image:
        if image_path not in self._image_cache:
            img = Image.open(image_path).convert("RGBA")
            img = img.resize(self.BASE_IMAGE_SIZE, Image.Resampling.LANCZOS)
            self._image_cache[image_path] = img
        return self._image_cache[image_path].copy()

    def _get_coordinates_for_single_plant(self):
        x_coord = self.BASE_IMAGE_SIZE[0] // 3 - self.SINGLE_IMAGE_SIZE[0] // 2 - self.OFFSET
        y_coord_step = self.SINGLE_IMAGE_SIZE[1] - (self.OFFSET * 2)

        coords = []
        for y in range(0, self.BASE_IMAGE_SIZE[1], y_coord_step):
            coords.append((x_coord, y))
            coords.append((x_coord + (self.SINGLE_IMAGE_SIZE[0] - (self.OFFSET * 2)), y - self.OFFSET))
            coords.append((x_coord + 2 * (self.SINGLE_IMAGE_SIZE[0] - (self.OFFSET * 2)), y - self.OFFSET))
        return coords

    def get_patch_of_trifoliate(self, disease="healthy") -> Image:

        angles = self.utility.get_random_angle(self.PLANT_PER_PATCH)
        coords = self._get_coordinates_for_single_plant()

        background_image = self._load_and_prepare_image(random.choice(self.background_paths))
        background_image = background_image.resize(self.BASE_IMAGE_SIZE)

        for (coord, angle) in zip(coords, angles):
            single_plant = self.single_plant.get_single_plant(disease)
            # single_plant = single_plant.rotate(angle)
            background_image.paste(single_plant, coord, single_plant)
        return background_image

if __name__ == '__main__':
    trifoliate_patch = TrifoliatePatch()

    start_time = time.time()
    trifoliate_patch.get_patch_of_trifoliate("healthy")
    end_time = time.time()
    print("Total time taken: ", end_time - start_time)