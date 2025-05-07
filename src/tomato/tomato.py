import math
from abc import ABC

import numpy as np
import random

from PIL import Image

from src.model.composite_leaf import CompositeLeaf
from src.model.plant import Plant
from src.tomato.tomato_branch import TomatoBranch


class Tomato(Plant):
    def __init__(self, config):
        self.config = config
        self.bifoliate = CompositeLeaf(config)
        self.branch_gen = TomatoBranch(config)
        self.branch_image_size = 256
        self.branch_range = (6,10)
        self.__setup__()

    def __setup__(self):
        self.divergence_angle = 137.5
        self.BASE_LEAF_SIZE = 50
        self.LEAF_SCALE_FACTOR = 40

    def get_leaf_arrangement_coords(self, center, num_leaves=25):
        num_branch = random.randint(self.branch_range[0], self.branch_range[1])
        # take randon angle
        divergence_angle = 137.5
        scaling_factor = 50
        # Convert angle to radians
        angle_rad = np.radians(divergence_angle)

        coords = []
        sizes = []

        for i in range(num_branch):
            # Calculate angle
            theta = i * angle_rad

            # Calculate radius (using concentric circles approach)
            # r = scaling_factor * np.sqrt(i / (2 * np.pi))
            r = scaling_factor * np.sqrt(i)

            # Convert polar coordinates to Cartesian
            x_i = r * np.cos(theta) + center[0]
            y_i = r * np.sin(theta) + center[1]

            # now size based on distance
            # distance formula does not have center as the center coords will cancel out anyway
            distance = np.sqrt(x_i ** 2 + y_i ** 2)
            leaf_size = self.BASE_LEAF_SIZE + distance * self.LEAF_SCALE_FACTOR
            sizes.append(leaf_size)

            coords.append((int(x_i), int(y_i)))
        return coords

    def get_angle_of_rotation_for_coords(self, center, coords):
        angles = []
        angle = 0
        for i,coord in enumerate(coords):
            angle = (i * self.divergence_angle) % 360
            angles.append(angle)
        return angles

    def get_leaves(self, disease_type, angle, scale_factor=1):
        # calculate number of bifoliates
        num_bifoliates = random.randint(5,7)
        branch = self.branch_gen.get_branch_with_leaves(disease_type, num_bifoliates, scale_factor)
        branch = branch.resize((self.branch_image_size, self.branch_image_size))
        return branch
