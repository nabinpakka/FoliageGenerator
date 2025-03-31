from PIL import Image
import random
import os
import math


class SoybeanPatch:
    def __init__(self, leaf_path: str, bg_path: str):
        self.leaf_path = leaf_path
        self.bg_path = bg_path
        self.leaf_size = (50, 50)
        self.canvas_size = (500, 500)
        self.leaf_cache = {}

    def _load_leaf(self, path: str) -> Image.Image:
        if path not in self.leaf_cache:
            img = Image.open(path).convert("RGBA")
            img = img.resize(self.leaf_size, Image.Resampling.LANCZOS)
            self.leaf_cache[path] = img
        return self.leaf_cache[path].copy()

    def _create_compound_leaf(self, leaf_paths: list) -> Image.Image:
        compound = Image.new('RGBA', (150, 150), (0, 0, 0, 0))

        # Center leaflet
        center_leaf = self._load_leaf(random.choice(leaf_paths))
        compound.paste(center_leaf, (50, 0), center_leaf)

        # Side leaflets
        angles = [-30, 30]  # Natural angles for side leaflets
        for i, angle in enumerate(angles):
            leaf = self._load_leaf(random.choice(leaf_paths))
            leaf = leaf.rotate(angle, expand=True)
            compound.paste(leaf, (25 if i == 0 else 75, 50), leaf)

        return compound

    def generate_patch(self, disease_percent: int = 30) -> Image.Image:
        # Load images
        healthy_leaves = [os.path.join(self.leaf_path, "healthy", f)
                          for f in os.listdir(os.path.join(self.leaf_path, "healthy"))
                          if f.endswith('.png')]
        diseased_leaves = [os.path.join(self.leaf_path, "frogeye", f)
                           for f in os.listdir(os.path.join(self.leaf_path, "frogeye"))
                           if f.endswith('.png')]

        # Create canvas
        background = Image.open(self.bg_path).convert("RGBA")
        background = background.resize(self.canvas_size)

        # Generate compound leaves
        num_compounds = random.randint(15, 25)
        for _ in range(num_compounds):
            # Decide if compound leaf will be diseased
            leaf_paths = diseased_leaves if random.randint(0, 100) < disease_percent else healthy_leaves

            # Create and place compound leaf
            compound = self._create_compound_leaf(leaf_paths)

            # Random position and rotation
            x = random.randint(0, self.canvas_size[0] - 150)
            y = random.randint(0, self.canvas_size[1] - 150)
            angle = random.randint(0, 360)

            compound = compound.rotate(angle, expand=True)
            background.paste(compound, (x, y), compound)

        return background

patch_gen = SoybeanPatch("/Users/hubbleloo/Documents/ResearchAssistant/Disease Classification/soyabean/DiseaseClassification/src/images", "/Users/hubbleloo/Documents/ResearchAssistant/Disease Classification/soyabean/DiseaseClassification/src/background_images/cropped_images/crop_0005.png")
patch = patch_gen.generate_patch(disease_percent=30)
patch.show()
# patch.save("soybean_patch.png")