import random

import numpy as np

import math

from src.model.composite_leaf import CompositeLeaf
from src.model.plant import Plant


class Soybean(Plant):
    def __init__(self, config):
        self.config = config
        self.trifoliate = CompositeLeaf(config)
        self.branch_range = (4,8)
        pass

    def get_leaf_arrangement_coords(self, center, num_leaves=25):
        # Phyllotactic parameters
        golden_angle = np.radians(137.5)  # Golden angle in radians
        scaling_factor = 35  # Controls spacing between leaves

        # Calculate leaf positions
        angles = []
        radii = []

        for n in range(1, num_leaves):
            angle = n * golden_angle
            radius = scaling_factor * np.sqrt(n)

            angles.append(angle)
            radii.append(radius)

        # Convert to Cartesian coordinates
        xs = [center[0] + r * np.cos(theta) for r, theta in zip(radii, angles)]
        ys = [center[1] + r * np.sin(theta) for r, theta in zip(radii, angles)]

        # converting to tuple
        coords = []
        for x, y in zip(xs, ys):
            coords.append((int(x), int(y)))

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
        return self.trifoliate.get_trifoliate(disease_type, angle, scale_factor)
