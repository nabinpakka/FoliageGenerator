import json
import math
import random
from typing import Tuple, Dict, List

import numpy as np

class Utility:
    def get_phyllotactic_leaf_arrangement_coords(self, center, num_trifoliates=25):
        # Phyllotactic parameters
        golden_angle = np.radians(137.5)  # Golden angle in radians
        scaling_factor = 35  # Controls spacing between leaves
        vertical_scaling = 0.15  # Controls vertical growth
        
        # Calculate leaf positions
        angles = []
        radii = []

        for n in range(1, num_trifoliates):
            angle = n * golden_angle
            radius = scaling_factor * np.sqrt(n)

            angles.append(angle)
            radii.append(radius)

        
        # Convert to Cartesian coordinates
        xs =  [center[0] + r * np.cos(theta) for r, theta in zip(radii, angles)]
        ys =  [ center [1] + r * np.sin(theta) for r, theta in zip(radii, angles)]

        #converting to tuple
        coords = []
        for x, y in zip(xs, ys):
            coords.append((int(x),int(y)))

        return coords

    def __get_random_with_step(self, start, end, step=20):
        """
        Generate a random number in a range such that all values are step apart.
        """
        # Create a list of numbers separated by 'step'
        possible_numbers = list(range(start, end + 1, step))

        # Randomly choose one number
        return random.choice(possible_numbers)
    def get_random_coordinates(self, image_size: int, steps: Tuple[int, int], count=1):
        coords = []
        for i in range(0, count):
            coords.append((self.__get_random_with_step(500, 400, step=steps[0]),
                           self.__get_random_with_step(0, image_size, step=steps[1])))
        return coords

    def get_random_angle(self, count: int, step=10):
        angles = []
        for i in range(0, count):
            angles.append(self.__get_random_with_step(0, 360, step))
        return angles

    def get_circular_coordinates(self, center: Tuple[int, int], radius: int, num_points: int, image_size = 300) -> list:
        """
        Generate coordinates for placing images in a circle.

        Args:
            center_x: x coordinate of circle center
            center_y: y coordinate of circle center
            radius: radius of the circle
            num_points: number of points to generate

        Returns:
            List of (x,y) coordinate tuples
        """
        coordinates = []
        center_x = center[0]
        center_y = center[0]

        # Adjust radius to account for image size
        radius += image_size // 2

        for i in range(num_points):
            # Calculate angle for even distribution
            angle = 2 * math.pi * i / num_points

            # Convert polar to cartesian coordinates
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            # Adjust for image size and top-left origin
            x -= image_size // 2
            y -= image_size // 2

            coordinates.append((round(x), round(y)))

        return coordinates

    def json_parser(self, file_path) -> Dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def string_to_list(self, input, separator = ',') -> List:
        return [part.strip() for part in input.split(separator)]