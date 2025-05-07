from PIL import Image, ImageDraw
import numpy as np
from math import sqrt


def bezier_point(t, p0, p1, p2, p3):
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


def generate_control_points(start_point, end_point, curvature=0.5):
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


def draw_branch(draw, start_point, end_point, width=3, color=(160, 64, 0),
                steps=100, control_points=None, texture=True, seed=None):
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
        cp1, cp2 = generate_control_points(start_point, end_point)
    else:
        cp1, cp2 = control_points

    # Draw the main branch
    # Calculate points along the curve
    points = []
    for t in np.linspace(0, 1, steps):
        points.append(bezier_point(t, start_point, cp1, cp2, end_point))

    # Draw lines between adjacent points with the specified width
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=width)

    # Add texture/details if requested
    if texture and width > 2:
        # Darker shade for texture
        texture_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))

        # Add some small curved details along the branch
        num_details = max(1, int(width * 1.5))
        for _ in range(num_details):
            # Pick two nearby points on the curve
            idx = np.random.randint(0, len(points) - 10)
            detail_start = points[idx]
            detail_end = points[idx + np.random.randint(5, 10)]

            # Create a small detail curve
            detail_cp1, detail_cp2 = generate_control_points(detail_start, detail_end, curvature=1.0)

            # Draw the detail
            detail_points = []
            for t in np.linspace(0, 1, 20):
                detail_points.append(bezier_point(t, detail_start, detail_cp1, detail_cp2, detail_end))

            for i in range(len(detail_points) - 1):
                draw.line([detail_points[i], detail_points[i + 1]], fill=texture_color, width=1)


def draw_branches(coordinates, image_size=(800, 600), bg_color=(255, 255, 255),
                  branch_color=(160, 64, 0), min_width=2, max_width=10, seed=None):
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
    img = Image.new('RGB', image_size, bg_color)
    draw = ImageDraw.Draw(img)

    # Connect points with branches
    for i in range(len(coordinates) - 1):
        # Calculate branch width based on position in sequence (thicker at the base)
        width_factor = 1 - (i / (len(coordinates) - 1))
        branch_width = int(min_width + (max_width - min_width) * width_factor)

        draw_branch(
            draw=draw,
            start_point=coordinates[i],
            end_point=coordinates[i + 1],
            width=branch_width,
            color=branch_color,
            texture=(branch_width > 2)
        )

        # Draw a small circle at each coordinate point
        node_radius = max(2, branch_width // 2)
        for x, y in [coordinates[i], coordinates[i + 1]]:
            draw.ellipse(
                (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
                fill=(125, 40, 0)
            )

    return img


# Example usage
if __name__ == "__main__":
    # Sample coordinates to connect
    points = [
        (100, 100),
        (200, 150),
        (350, 120),
        (500, 200),
        (650, 150)
    ]

    # Alternative example with custom coordinates
    # You can also connect arbitrary sets of points
    branch_sets = [
        [(100, 100), (300, 50), (500, 150)],
        [(100, 300), (200, 400), (400, 350), (600, 400)]
    ]

    custom_img = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
    draw = ImageDraw.Draw(custom_img)

    for branch_set in branch_sets:
        for i in range(len(branch_set) - 1):
            draw_branch(
                draw=draw,
                start_point=branch_set[i],
                end_point=branch_set[i + 1],
                width=5,
                color=(85, 107, 47, 255)
            )

    custom_img.save("custom_branches.png")
    custom_img.show()