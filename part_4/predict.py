import argparse
import sys
import torch
import plotly.graph_objects as go
import numpy as np
from leaffliction.constants import (
    DEVICE,
    PATHS,
    DFLT_MODEL_KWARGS,
    ID2LABEL,
)
from leaffliction.models import CNN
from leaffliction.utils import load_image


def process_image(img_path: str) -> torch.Tensor:
    img = torchvision.io.decode_image(p)
    img = img.permute(1, 2, 0) / 255
    return img


def load_model(model_path: str) -> CNN:
    """Load trained model from checkpoint."""
    model = CNN(**DFLT_MODEL_KWARGS).to(device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict(model: CNN, img: torch.Tensor) -> tuple[str, float]:
    """Make prediction on preprocessed image."""
    with torch.no_grad():
        with torch.autocast(DEVICE.type, torch.bfloat16):
            output = model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_id = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_id].item()
    
    predicted_label = ID2LABEL[predicted_id]
    return predicted_label, confidence


def tensor_to_numpy_for_display(img: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for plotly display."""
    # Handle different tensor formats
    if img.dim() == 4:  # (B, C, H, W)
        img = img[0]  # Remove batch dimension
    if img.dim() == 3:  # (C, H, W)
        img = img.permute(1, 2, 0)  # Convert to (H, W, C)
    
    # Convert to numpy and handle dtype
    if img.dtype == torch.bfloat16:
        img = img.float()
    img_np = img.cpu().numpy()
    
    # Normalize to [0, 255] for plotly (uint8 range)
    if img_np.max() <= 1.0:
        img_np = img_np * 255.0
    
    # Clamp to [0, 255] and convert to uint8
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


def display_results(original_img: torch.Tensor, predicted_label: str, confidence: float):
    """Display original image and prediction."""
    # Convert image to numpy array
    original_np = tensor_to_numpy_for_display(original_img)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add image as a heatmap-like trace
    fig.add_trace(
        go.Image(z=original_np)
    )
    
    # Update layout with title
    title_text = f"Predicted Disease: {predicted_label}<br>Confidence: {confidence:.2%}"
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
        height=800,
        width=800,
    )
    
    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a leaf image and predict the disease type"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PATHS["model"],
        help=f"Path to the trained model file (default: {PATHS['model']})"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    import os
    if not os.path.exists(args.image_path):
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        sys.exit(1)
    
    # original_img = load_image(args.image_path)
    # model = load_model(args.model)
    processed_img = process_image(args.image_path)
    print(processed_img)
    # predicted_label, confidence = predict(model, preprocessed_img)
    # display_results(original_img, predicted_label, confidence)


if __name__ == "__main__":
    main()

