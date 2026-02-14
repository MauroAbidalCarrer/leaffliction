import os
import argparse
import torch
from torch import Tensor
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from leaffliction.constants import (
    DEVICE,
    DFLT_MODEL_KWARGS,
    ID2LABEL,
    PATHS
)
from leaffliction.models import CNN
from leaffliction.utils import load_image_as_tensor


def process_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return np.array(img)

def load_model(model_path: str) -> CNN:
    model = CNN(**DFLT_MODEL_KWARGS).to(device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def predict(model: CNN, img_path: str) -> tuple[str, float]:
    img_t : Tensor = load_image_as_tensor(img_path)
    img_t = img_t.unsqueeze(0).to(device=DEVICE).float()
    with torch.no_grad():
        with torch.autocast(DEVICE.type, torch.bfloat16):
            output : Tensor = model(img_t)
            probabilities : Tensor = torch.nn.functional.softmax(output, dim=1)
            predicted_id : int = output.argmax(dim=1).item()
            confidence : float = probabilities[0][predicted_id].item()
    predicted_label : str = ID2LABEL[predicted_id]
    return predicted_label, confidence

def display_prediction(original_img: np.ndarray, predicted_label: str, confidence: float):
    title_text = f"Predicted Disease: {predicted_label}<br>Confidence: {confidence:.2%}"
    fig = go.Figure()
    fig.add_trace(go.Image(z=original_img))
    fig.update_layout(
        width=256,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=title_text, x=0.5, xanchor="center")
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    image_to_predict_path = args.image_path
    if not os.path.exists(image_to_predict_path):
        print(f"Error: Path does not exist: {image_to_predict_path}")
        return

    try:
        model = load_model(PATHS['model'])
        original_img = process_image(image_to_predict_path)
        predicted_label, confidence = predict(model, image_to_predict_path)
        display_prediction(original_img, predicted_label, confidence)
    except Exception as e:
        print(f'An error has occured: {e}')

if __name__ == "__main__":
    main()
