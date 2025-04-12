from functools import lru_cache

from leaf import  Leaf
from PIL import Image

import os


class CompositeLeaf(Leaf):
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