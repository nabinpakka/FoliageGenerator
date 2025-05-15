import os
import random
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw

from src.model.composite_leaf import CompositeLeaf
from src.utilities import Utility


class TomatoBranch:
    def __init__(self, config):
        self._image_cache = {}
        self.branch_image_size = (200, 600)
        self.leaves = CompositeLeaf(config)
        self.bifoliate_offset = 50
        self.leaf_spacing = config.get("leaf_spacing")
        self.leaf_size = config.get("single_leaf_size")

    def __get_increasing_random_num(self, current, offset):
        return random.randint(current, current + offset)

    def get_branch_with_leaves(self, disease, num_bifoliates, scaling_factor) -> Image:

        background_image_height = (num_bifoliates + 1) * self.leaf_size

        size_of_bifoliate_image = self.leaf_size * 4
        background = Image.new("RGBA", (size_of_bifoliate_image, background_image_height), (0, 0, 0, 0))

        x_coord_base = self.branch_image_size[0] // 2 - self.leaf_size - self.leaf_spacing // 4

        branch_coords = []
        scaled_size = 0
        current_offset = 0
        # where should the branch lean
        # true for right
        random_lean = random.randint(0, 1)

        for i in range(num_bifoliates):
            bifoliate = self.leaves.get_bifoliate(disease)

            y_coord = background_image_height - (i + 1) * self.leaf_size + self.bifoliate_offset
            current_offset = self.__get_increasing_random_num(current_offset, 10)
            if random_lean:
                # lean right
                x_coord = x_coord_base + current_offset
            else:
                # lean left
                x_coord = x_coord_base - current_offset

            # resizing image as we move up the branch
            scaling_factor += i * 0.05
            scaled_size = int(self.leaf_size * scaling_factor * 2)

            # calculate branch_coords
            branch_x_coord = x_coord + scaled_size // 2 + int(i * scaling_factor * 5)
            branch_y_coord = y_coord + self.leaf_size

            # adding initial coord
            if not i:
                branch_coords.append((branch_x_coord + 15, branch_y_coord + self.leaf_size))
            branch_coords.append((branch_x_coord + 15, branch_y_coord))

            bifoliate = bifoliate.resize((scaled_size, scaled_size))
            background.paste(bifoliate, (x_coord, y_coord), bifoliate)

        background = self.draw_branches(branch_coords, background, branch_color=(85, 107, 47))

        # add a leaf at the top of the branch
        coord = branch_coords[-1]
        top_leaf = self.leaves.get_single_leaf(disease)
        top_leaf = top_leaf.resize((scaled_size // 2, scaled_size // 2))
        background.paste(top_leaf, (coord[0] - self.leaf_size // 2 - current_offset, coord[1] - self.leaf_size // 2),
                         top_leaf)
        return background

    def bezier_point(self, t, p0, p1, p2, p3):
        """
        Calculate a point on a cubic Bezier curve.

        Parameters:
        - t: parameter between 0 and 1
        - p0: start point (x, y)
        - p1, p2: control points (x, y)
        - p3: end point (x, y)

        Returns:
        - (x, y) tuple of coordinates on the curve
        """
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        return (x, y)

    def generate_control_points(self, start_point, end_point, curvature=0.5):
        """
        Generate control points for a Bezier curve between two points.

        Parameters:
        - start_point: (x, y) tuple
        - end_point: (x, y) tuple
        - curvature: how curved the line should be (0.0-1.0)

        Returns:
        - Two (x, y) control points
        """
        # Vector from start to end
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Distance between points
        length = sqrt(dx * dx + dy * dy)

        # Control point distances (1/3 and 2/3 along the path)
        cp1_x = start_point[0] + dx / 3
        cp1_y = start_point[1] + dy / 3
        cp2_x = start_point[0] + 2 * dx / 3
        cp2_y = start_point[1] + 2 * dy / 3

        # Add some randomness to control points for natural look
        # Perpendicular vector
        perp_dx = -dy
        perp_dy = dx

        # Normalize and scale by curvature and length
        magnitude = sqrt(perp_dx * perp_dx + perp_dy * perp_dy)
        if magnitude > 0:
            perp_dx = perp_dx / magnitude * length * curvature
            perp_dy = perp_dy / magnitude * length * curvature

        # Apply perpendicular offset in opposite directions
        direction = 1 if np.random.random() > 0.5 else -1
        cp1_x += direction * perp_dx * np.random.uniform(0.2, 0.8)
        cp1_y += direction * perp_dy * np.random.uniform(0.2, 0.8)
        cp2_x += -direction * perp_dx * np.random.uniform(0.2, 0.8)
        cp2_y += -direction * perp_dy * np.random.uniform(0.2, 0.8)

        return (cp1_x, cp1_y), (cp2_x, cp2_y)

    def draw_branch(self, draw, start_point, end_point, width=2, color=(85, 107, 47),
                    steps=3, control_points=None, texture=True, seed=None):
        """
        Draw a branch with curves from one point to another.

        Parameters:
        - draw: ImageDraw object
        - start_point: (x, y) tuple for starting point
        - end_point: (x, y) tuple for ending point
        - width: width of the branch
        - color: RGB tuple for branch color (tomato branch color by default)
        - steps: number of points to use for the curve (more = smoother)
        - control_points: optional list of two control points, will be generated if None
        - texture: whether to add texture details to the branch
        - seed: random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        if control_points is None:
            cp1, cp2 = self.generate_control_points(start_point, end_point)
        else:
            cp1, cp2 = control_points

        # Draw the main branch
        # Calculate points along the curve
        points = []
        for t in np.linspace(0, 1, steps):
            points.append(self.bezier_point(t, start_point, cp1, cp2, end_point))

        # Draw lines between adjacent points with the specified width
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=width)

        #draw lines for petiole
        petiole_x, petiole_y = start_point
        draw.line([start_point, (petiole_x +20, petiole_y - 10)], fill=color, width=width //2 + 3)
        draw.line([start_point, (petiole_x - 20, petiole_y -10 )],  fill=color, width=width//2  + 3)

    def draw_branches(self, coordinates, img,
                      branch_color=(85, 107, 47), min_width=2, max_width=10, seed=None):
        """
        Draw branches connecting a list of coordinates.

        Parameters:
        - coordinates: list of (x, y) tuples to connect
        - image_size: (width, height) of the output image
        - bg_color: RGB background color
        - branch_color: RGB color for branches
        - min_width, max_width: range for branch width
        - seed: random seed for reproducibility

        Returns:
        - PIL Image object
        """
        if seed is not None:
            np.random.seed(seed)

        # Create a blank image
        draw = ImageDraw.Draw(img)

        # Connect points with branches
        for i in range(len(coordinates) - 1):
            # Calculate branch width based on position in sequence (thicker at the base)
            width_factor = 1 - (i / (len(coordinates) - 1))
            branch_width = int(min_width + (max_width - min_width) * width_factor)

            self.draw_branch(
                draw=draw,
                start_point=coordinates[i],
                end_point=coordinates[i + 1],
                width=branch_width,
                color=branch_color,
                texture=(branch_width > 2)
            )
        return img


if __name__ == '__main__':
    pass

    utility = Utility()
    config = utility.json_parser(
        "path_to_config_file")
    tomato_branch = TomatoBranch(config)

    for i in range(3, 6):
        tomato_branch.get_branch_with_leaves("target_spot", i, 1)
