import math
from abc import ABC

import numpy as np
import random

from src.model.composite_leaf import CompositeLeaf
from src.model.plant import Plant


class Tomato(Plant):
    def __init__(self, config):
        self.config = config
        self.bifoliate = CompositeLeaf(config)
        self.__setup__()

    def __setup__(self):
        self.patterns = [133, 135, 137]
        self.BASE_LEAF_SIZE = 50
        self.LEAF_SCALE_FACTOR = 40

    def get_leaf_arrangement_coords(self, center, num_leaves=25):
        # take randon angle
        divergence_angle = random.choice(self.patterns)
        scaling_factor = 40
        # Convert angle to radians
        angle_rad = np.radians(divergence_angle)

        coords = []
        sizes = []

        for i in range(num_leaves):
            # Calculate angle
            theta = i * angle_rad

            # Calculate radius (using concentric circles approach)
            # r = scaling_factor * np.sqrt(i / (2 * np.pi))
            r = scaling_factor * np.sqrt(i )

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
        x2, y2 = center
        for coord in coords:
            x1, y1 = coord

            dx = x2 - x1
            dy = y2 - y1

            if dy == 0:
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

    def get_leaves(self, disease_type, angle, scale_factor=1):
        return self.bifoliate.get_bifoliate(disease_type, angle, scale_factor )
