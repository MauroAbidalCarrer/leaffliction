import torchvision
import torch
from torch import Tensor

def load_image_as_tensor(image_path: str) -> Tensor:
    """Load and decode image from path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image tensor with shape (C, H, W) and dtype uint8
    """
    img_t = torchvision.io.decode_image(image_path).to(dtype=torch.uint8)
    return img_t