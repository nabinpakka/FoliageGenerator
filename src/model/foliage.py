import os
import random
import time
from functools import lru_cache

from PIL import Image
from typing import Dict

from src.model.single_plant import SinglePlant
from src.utilities import Utility


# TODO parse config file

class Foliage:
    def __init__(self, config):

        self.PLANT_PER_PATCH = config.get("num_plants")
        self.BASE_IMAGE_SIZE = eval(config.get("foliage_size"))
        self.SINGLE_IMAGE_SIZE = config.get("single_plant_size")
        self.OFFSET = config.get("plant_offset")
        self.TYPE = config.get("type")
        self.config = config
        self.BASE_BACKGROUND_IMAGE = config.get("background_image_path")
        self._setup()

    def _setup(self):
        self._image_cache: Dict[str, Image.Image] = {}
        self.single_plant = SinglePlant(self.config)
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

        column_gap = 2
        if self.TYPE == "tomato":
            column_gap = 2

        x_coord = self.BASE_IMAGE_SIZE[0] // 3 - self.SINGLE_IMAGE_SIZE // 2 - self.OFFSET
        y_coord_step = self.SINGLE_IMAGE_SIZE - (self.OFFSET * 2)

        coords = []
        for y in range(0, self.BASE_IMAGE_SIZE[1], y_coord_step):
            coords.append((x_coord, y))
            coords.append((x_coord + (self.SINGLE_IMAGE_SIZE - int(self.OFFSET * column_gap)), y - self.OFFSET))
            coords.append((x_coord + 2 * (self.SINGLE_IMAGE_SIZE - int(self.OFFSET * column_gap)), y - self.OFFSET))
        return coords

    def get_patch_indices(self, cluster_id, rows=4, cols=3, patch_size=2):
        """
        Returns the flat indices of a patch in a 1D array representing a 2D grid.

        Parameters:
            cluster_id (int): Number of the patch (starting from 1)
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            patch_size (int): Size of the square patch (default is 2 for 2x2)

        Returns:
            List[int]: Flat indices of the patch elements
        """
        max_row_start = rows - patch_size
        max_col_start = cols - patch_size
        total_patches = (max_row_start + 1) * (max_col_start + 1)

        if cluster_id < 1 or cluster_id > total_patches:
            raise ValueError(f"Patch number must be between 1 and {total_patches}")

        # Map patch_number to its (row, col) starting position

        row_id = (cluster_id - 1) // (max_col_start + 1)
        col_id = (cluster_id - 1) % (max_col_start + 1)

        indices = []
        for dy in range(patch_size):
            for dx in range(patch_size):
                r = row_id + dy
                c = col_id + dx
                flat_index = r * cols + c
                indices.append(flat_index)

        return indices

    def get_cluster_coords_index(self):
        cluster_id = random.randint(1, 6)

        # dividing the foliage image to 6 regions to with 4 single plants
        # for i in range(6):
        #     coord_index = [(cluster_id * 3 + i) for i in range(0,6)]

        return self.get_patch_indices(cluster_id)

        #get rand
    def get_patch_of_leaves(self, disease="healthy") -> Image:

        # angles = self.utility.get_random_angle(self.PLANT_PER_PATCH)
        coords = self._get_coordinates_for_single_plant()
        cluster_coord_index = self.get_cluster_coords_index()

        background_image = self._load_and_prepare_image(random.choice(self.background_paths))
        background_image = background_image.resize(self.BASE_IMAGE_SIZE)

        for idx, coord in enumerate(coords):
            if idx in cluster_coord_index:
                single_plant = self.single_plant.get_single_plant(disease)
            else:
                single_plant = self.single_plant.get_single_plant("healthy")
            # single_plant = single_plant.rotate(angle)
            background_image.paste(single_plant, coord, single_plant)
        return background_image

if __name__ == '__main__':

    utility = Utility()
    config =  utility.json_parser("/Users/roshan/Documents/ResearchAssistant/DiseaseClassification/FoliageGenerator/src/soybean/config.json")
    trifoliate_patch = Foliage(config)

    start_time = time.time()
    patch = trifoliate_patch.get_patch_of_leaves("bacterial_blight")
    patch.show()
    end_time = time.time()
    print("Total time taken: ", end_time - start_time)