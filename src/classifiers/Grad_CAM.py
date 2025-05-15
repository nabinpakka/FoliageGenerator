import numpy as np
import paddle
import paddle.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class Display:
    def __init__(self, model, layer):
        """
        Initialize the Display class for attention visualization

        Args:
            model: The neural network model
            layer: The target layer for visualization
        """
        self.model = model
        self.layer = layer
        self.gradient = None
        self.activation = None

        # Register hooks
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to the target layer"""

        def forward_hook(layer, input, output):
            self.activation = output

            # Define gradient hook within forward hook
            def backward_hook(grad):
                self.gradient = grad
                return grad

            output.register_hook(backward_hook)

        # Register only forward hook
        self.hook_handles.append(self.layer.register_forward_post_hook(forward_hook))

    def _release_hooks(self):
        """Remove the registered hooks"""
        for handle in self.hook_handles:
            handle.remove()

    def _preprocess_image(self, img):
        """
        Preprocess the input image for model inference

        Args:
            img: Input image (RGB format)

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Define transforms
        transform = paddle.vision.transforms.Compose([
            paddle.vision.transforms.Resize(size=512),
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transforms and add batch dimension
        img_tensor = transform(img)
        img_tensor = paddle.unsqueeze(img_tensor, axis=0)
        return img_tensor

    def _generate_cam(self, input_tensor):
        """
        Generate Class Activation Map

        Args:
            input_tensor: Preprocessed input image tensor

        Returns:
            cam: Generated attention map
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Get the score for the predicted class
        score = output.max()

        # Backward pass
        self.model.clear_gradients()
        score.backward()

        # Generate CAM
        weights = paddle.mean(self.gradient, axis=(2, 3))
        cam = paddle.zeros(self.activation.shape[2:])

        for i, w in enumerate(weights[0]):
            cam += w * self.activation[0, i]

        cam = paddle.maximum(cam, paddle.to_tensor(0.))
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.numpy()

    def save(self, img, file='attention_map.jpg'):
        """
        Generate and save the attention visualization

        Args:
            img: Input image (RGB format)
            file: Output file path
        """
        # Preprocess image
        input_tensor = self._preprocess_image(img)

        # Generate CAM
        cam = self._generate_cam(input_tensor)

        # Resize CAM to original image size
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Superimpose heatmap on original image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        # Create figure with subplots
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')

        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.title('Attention Map')
        plt.axis('off')

        # Superimposed
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        plt.title('Superimposed')
        plt.axis('off')

        # Save figure
        plt.tight_layout()
        plt.savefig(file)
        plt.close()

    def __del__(self):
        """Cleanup: remove hooks when object is deleted"""
        self._release_hooks()

    @staticmethod
    def show_network(model):
        """
        Print model architecture

        Args:
            model: The neural network model
        """
        print("Model Architecture:")
        print(model)