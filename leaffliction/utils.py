import torchvision
from leaffliction.constants import DEVICE
from torch import Tensor

def load_image_as_tensor(image_path: str) -> Tensor:
    """Load and decode image from path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image tensor with shape (C, H, W) and dtype uint8
    """
    img = torchvision.io.decode_image(image_path).to(DEVICE)
    return img

