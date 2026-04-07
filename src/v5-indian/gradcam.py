"""
gradcam.py
──────────
Grad-CAM visualisation — shows which retinal regions the model
focused on when making its prediction.

Usage:
  from gradcam import run_gradcam
  run_gradcam(model, val_df, sample_idx=5)
"""

import cv2
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

from config import IMAGE_DIR, RESULTS_DIR, IMG_SIZE
from train import device

GRADE_LABELS = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer and computes a heatmap
    showing which spatial regions most influenced the prediction.

    Used for clinical explainability — lets doctors see what the
    model is 'looking at' when it grades a retinal image.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates CAM heatmap for the given class index.

        Args:
            image_tensor : (3, H, W) float tensor
            class_idx    : target class (DR grade 0–4)

        Returns:
            cam : (H, W) numpy array, values in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Forward pass
        dr_out, _ = self.model(image_tensor, torch.zeros(1, 512).to(device))
        self.model.zero_grad()

        # Backward pass for target class
        dr_out[0, class_idx].backward()

        # Weighted combination of activation maps
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def run_gradcam(model, val_df, sample_idx: int = 0):
    """
    Runs Grad-CAM on a single validation image and saves the result.

    Args:
        model      : trained ConvNextModel or MultimodalModel
        val_df     : validation DataFrame
        sample_idx : which sample to visualise
    """

    # Hook into last conv block of ConvNext backbone
    target_layer = model.backbone.features[-1]
    gradcam      = GradCAM(model, target_layer)

    # Load sample
    row      = val_df.iloc[sample_idx]
    img_path = Path(IMAGE_DIR) / row['image']
    image    = torchvision.io.read_image(str(img_path)).float() / 255.0
    image    = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)

    true_label = int(row['level'])
    cam        = gradcam.generate(image, true_label)

    # Visualise
    orig    = image.permute(1, 2, 0).numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * orig + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f'Grad-CAM | True: Grade {true_label} ({GRADE_LABELS[true_label]})',
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(orig);    axes[0].set_title('Original');  axes[0].axis('off')
    axes[1].imshow(heatmap); axes[1].set_title('Grad-CAM');  axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title('Overlay');   axes[2].axis('off')

    plt.tight_layout()
    save_path = RESULTS_DIR / f'gradcam_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Grad-CAM saved → {save_path}')
    plt.show()
