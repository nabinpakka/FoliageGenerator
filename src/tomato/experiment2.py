import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_phyllotaxis(num_leaves=13, divergence_angle=137.5):
    """
    Generate points following a phyllotaxis pattern.

    Parameters:
    - num_leaves: Number of leaves to generate
    - divergence_angle: Angle between consecutive leaves in degrees

    Returns:
    - x, y: Coordinates of the points
    """
    # Convert angle to radians
    angle_rad = np.radians(divergence_angle)

    x = []
    y = []
    radii = []

    for i in range(num_leaves):
        # Calculate angle
        theta = i * angle_rad

        # Calculate radius (using concentric circles approach)
        r = np.sqrt(i / (2 * np.pi))
        # Round to nearest 0.5 to create distinct concentric rings
        r_rounded = np.round(r * 2) / 2

        # Convert polar coordinates to Cartesian
        x_i = r * np.cos(theta)
        y_i = r * np.sin(theta)

        x.append(x_i)
        y.append(y_i)
        radii.append(r_rounded)

    print(x)
    print(y)
    return x, y, list(set(radii))


def main():
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Create the three different patterns
    patterns = [
        {"angle": 133, "title": "Tomato Phyllotaxis (133°)", "ax": axes[0]},
        {"angle": 135, "title": "Tomato Phyllotaxis (135°)", "ax": axes[1]},
        {"angle": 137, "title": "Tomato Phyllotaxis (137°)", "ax": axes[2]}
    ]

    # Number of leaves to generate for each pattern
    num_leaves = 40

    # Create a green colormap for tomato leaf shades
    cmap = plt.cm.Greens

    for pattern in patterns:
        ax = pattern["ax"]
        angle = pattern["angle"]

        # Generate points
        x, y, unique_radii = generate_phyllotaxis(num_leaves, angle)

        # Draw concentric circles
        for radius in unique_radii:
            if radius > 0:  # Skip the center point
                circle = plt.Circle((0, 0), radius, fill=False, color='darkgreen',
                                    linewidth=0.5, alpha=0.3)
                ax.add_patch(circle)

        # Scale dot sizes based on position
        sizes = []
        colors = []

        for i in range(len(x)):
            # Calculate distance from center
            distance = np.sqrt(x[i] ** 2 + y[i] ** 2)

            # Leaf size increases with distance from center
            # Similar to how tomato leaves grow larger as they mature
            leaf_size = 20 + distance * 70
            sizes.append(leaf_size)

            # Vary color intensity based on position (simulating leaf maturity)
            # Newer leaves (higher index) have different color
            color_intensity = 0.5 + (i / num_leaves) * 0.4
            colors.append(cmap(color_intensity))

        # Plot leaves as green dots with varying sizes
        ax.scatter(x, y, s=sizes, c=colors, edgecolors='darkgreen',
                   linewidths=0.5, alpha=0.8, zorder=3)

        # Add stem at center
        ax.add_patch(plt.Circle((0, 0), 0.15, color='saddlebrown', zorder=2))

        ax.set_title(pattern["title"], fontsize=12)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Set limits
        max_radius = max(unique_radii) * 1.1
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Set background color to very light green
        ax.set_facecolor('#f5fff5')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()