import math
from functools import lru_cache
import random
from typing import Tuple, List, Dict

from PIL import Image

import os

from src.utilities import Utility


class CompositeLeaf():
    def __init__(self, config):
        self.images_count = {}
        self.image_path_by_disease = {}

        self.PLANT_PER_PATCH = config.get("num_plants")
        self.BASE_IMAGE_SIZE = config.get("foliage_size")
        self.LEAF_IMAGE_SIZE = config.get("single_leaf_size")
        self.BASE_BACKGROUND_IMAGE = config.get("background_image_path")
        self.BASE_LEAF_IMAGE_PATH = config.get("input_path")
        self.LEAF_SPACING = config.get("leaf_spacing")
        self._setup()

    def _setup(self):
        self._image_cache: Dict[str, Image.Image] = {}
        self.utility = Utility()
        self.__get_list_of_image_paths_by_disease()


    # cache some images for faster access
    @lru_cache(maxsize=1024)
    def _load_and_prepare_image(self, image_path: str) -> Image.Image:
        if image_path not in self._image_cache:
            img = Image.open(image_path).convert("RGBA")
            img = img.resize(self.LEAF_IMAGE_SIZE, Image.Resampling.LANCZOS)
            self._image_cache[image_path] = img
        return self._image_cache[image_path].copy()

    def __get_image_names_and_count_from_directory(self, base_path, dir_name="") -> Tuple[List[str], int]:
        image_paths = []
        image_count = 0
        dir_path = os.path.join(base_path, dir_name)
        for image_name in os.listdir(dir_path):
            if image_name.endswith('.png'):
                image_paths.append(os.path.join(dir_path, image_name))
                image_count += 1
        return image_paths, image_count

    def __get_list_of_image_paths_by_disease(self):
        dir_names = []
        for dir_name in os.listdir(self.BASE_LEAF_IMAGE_PATH):
            dir_path = os.path.join(self.BASE_LEAF_IMAGE_PATH, dir_name)
            if os.path.isdir(dir_path):
                dir_names.append(dir_name)

        for dir_name in dir_names:
            image_paths, image_count = self.__get_image_names_and_count_from_directory(self.BASE_LEAF_IMAGE_PATH,
                                                                                       dir_name)
            self.image_path_by_disease[dir_name] = image_paths
            self.images_count[dir_name] = image_count

    def __get_random_leaf_image_path_from_dir(self, disease):
        num_images = self.images_count[disease]
        random_image_index = random.randint(0, num_images - 1)
        random_image_path = self.image_path_by_disease[disease][random_image_index]

        # get random image from the directory
        return random_image_path

    def _get_leaves_image_paths(self, disease="healthy", num_leaves = 3):
        # generate random number for the number of healthy leaves to include in the trifoliate
        num_healthy_leaves = num_leaves if disease == "healthy" else random.randint(0, 2)

        leaf_image_paths = []
        leaf_image_paths.extend(random.sample(self.image_path_by_disease["healthy"], num_healthy_leaves))
        if disease != "healthy":
            leaf_image_paths.extend(random.sample(self.image_path_by_disease[disease], num_leaves - num_healthy_leaves))

        return leaf_image_paths

    def get_bifoliate(self, disease="healthy", angle=0, scale_factor=1) -> Image.Image:
        leaf_image_paths = self._get_leaves_image_paths(disease, 2)

        size_of_bifoliate_image = self.LEAF_IMAGE_SIZE * 2
        scaled_size = self.LEAF_IMAGE_SIZE * scale_factor

        leaf_image = Image.open(leaf_image_paths[0]).convert("RGBA")
        leaf_image1 = Image.open(leaf_image_paths[1]).convert("RGBA")

        background = Image.new("RGBA", (size_of_bifoliate_image, size_of_bifoliate_image), (0, 0, 0, 0))

        leaf_image = leaf_image.resize(
            (int(scaled_size), int(scaled_size)),
            Image.Resampling.LANCZOS)
        leaf_image1 = leaf_image1.resize(
            (int(scaled_size), int(scaled_size)),
            Image.Resampling.LANCZOS)

        rotated_image = leaf_image.rotate(-90)
        rotated_image1 = leaf_image1.rotate(90)

        half = size_of_bifoliate_image // 2
        x_offset = half
        y_offset = half
        x_offset1 = half - self.LEAF_SPACING
        y_offset1 = half

        background.paste(rotated_image, (x_offset, y_offset), leaf_image)
        background.paste(rotated_image1, (x_offset1, y_offset1), rotated_image1)
        background = background.rotate(angle)
        return background

    def get_trifoliate(self, disease="healthy", angle=0, scale_factor=1) -> Image.Image:
        # leaf_image_dir = os.path.join(BASE_LEAF_IMAGE_PATH, disease)
        leaf_image_paths = self._get_leaves_image_paths(disease)

        size_of_trifoliate_image = self.LEAF_IMAGE_SIZE * 2
        leaf_image = Image.open(leaf_image_paths[0]).convert("RGBA")
        leaf_image1 = Image.open(leaf_image_paths[1]).convert("RGBA")
        leaf_image2 = Image.open(leaf_image_paths[2]).convert("RGBA")


        background = Image.new("RGBA", (size_of_trifoliate_image, size_of_trifoliate_image), (0, 0, 0, 0))

        angle1 = 90
        angle2 = 180

        half = size_of_trifoliate_image // 2
        x_offset = half
        y_offset = half
        x_offset1 = half - self.LEAF_SPACING
        y_offset1 = half - self.LEAF_SPACING
        x_offset2 = half - math.ceil(1.5 * self.LEAF_SPACING)
        y_offset2 = half

        scaled_size = self.LEAF_IMAGE_SIZE * scale_factor

        leaf_image = leaf_image.resize(
            (int(scaled_size), int(scaled_size)),
            Image.Resampling.LANCZOS)
        leaf_image1 = leaf_image1.resize(
            (int(scaled_size), int(scaled_size)),
            Image.Resampling.LANCZOS)
        leaf_image2 = leaf_image2.resize(
            (int(scaled_size), int(scaled_size)),
            Image.Resampling.LANCZOS)

        rotated_image1 = leaf_image1.rotate(angle1)
        rotated_image2 = leaf_image2.rotate(angle2)

        background.paste(leaf_image, (x_offset, y_offset), leaf_image)
        background.paste(rotated_image1, (x_offset1, y_offset1), rotated_image1)
        background.paste(rotated_image2, (x_offset2, y_offset2), rotated_image2)
        background = background.rotate(angle)

        return background
