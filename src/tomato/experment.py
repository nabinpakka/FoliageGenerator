import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

# Set random seed for reproducibility
np.random.seed(42)


def generate_phyllotaxis(num_leaves=100, divergence_angle=137.5, scale_factor=0.8):
    """
    Generate points following a phyllotaxis pattern.

    Parameters:
    - num_leaves: Number of leaves to generate
    - divergence_angle: Angle between consecutive leaves in degrees
    - scale_factor: Growth factor for the radius

    Returns:
    - x, y: Coordinates of the points
    """
    # Convert angle to radians
    angle_rad = np.radians(divergence_angle)

    x = []
    y = []

    for i in range(num_leaves):
        # Calculate radius (distance from center)
        r = scale_factor * np.sqrt(i)

        # Calculate angle
        theta = i * angle_rad

        # Convert polar coordinates to Cartesian
        x_i = r * np.cos(theta)
        y_i = r * np.sin(theta)

        x.append(x_i)
        y.append(y_i)

    return x, y


def main():
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate phyllotaxis pattern
    num_leaves = 200
    x, y = generate_phyllotaxis(num_leaves=num_leaves, divergence_angle=137.5, scale_factor=0.35)

    # Draw stem (center of the plant)
    circle = plt.Circle((0, 0), 0.2, color='brown')
    ax.add_patch(circle)

    # Scale dot sizes based on distance from center (simulating leaf growth)
    sizes = []
    colors = []
    cmap = plt.cm.Greens

    for i in range(len(x)):
        # Leaf size increases with distance from center
        # Newer leaves (higher index) are larger
        leaf_size = 10 + i * 0.4
        sizes.append(leaf_size)

        # Color varies slightly based on position (simulating leaf maturity)
        # Normalize between 0.6 and 0.9 to get nice green shades
        color_intensity = 0.6 + (i / num_leaves) * 0.3
        colors.append(cmap(color_intensity))

    # Plot points representing the leaves with scaled sizes
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.8, edgecolors='darkgreen', linewidths=0.5)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits
    limit = max(max(abs(val) for val in x), max(abs(val) for val in y)) * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Set title
    plt.title('Tomato Plant Phyllotaxis Pattern (Scaled Dots as Leaves)', fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()