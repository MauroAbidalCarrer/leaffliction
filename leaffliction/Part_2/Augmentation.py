from PIL import Image
from pathlib import Path
import PIL
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import v2, CenterCrop
import logging

class Crop(CenterCrop):
    def __init__(self, resize_target=256, crop_ratio=0.80):
        """
        Initialize Crop transform with parameters for computing crop size.
        
        Args:
            resize_target: Target size for the smaller dimension after resize (default: 256)
            crop_ratio: Ratio of crop size to resize target (default: 0.80)
        """

        super().__init__(size=1)
        self.resize_target = resize_target
        self.crop_ratio = crop_ratio
    
    @staticmethod
    def _compute_crop_params(width, height, resize_target=256, crop_ratio=0.80):
        """
        Compute resize and crop parameters based on image dimensions.
        
        Args:
            width: Original image width
            height: Original image height
            resize_target: Target size for the smaller dimension after resize (default: 256)
            crop_ratio: Ratio of crop size to resize target (default: 0.80)
            
        Returns:
            crop_size: Size for CenterCrop transform (int or tuple)
        """
        crop_size = int(resize_target * crop_ratio)
        return crop_size
    
    def __call__(self, img):
        """
        Override __call__ to compute crop size dynamically from image dimensions.
        
        Args:
            img: PIL Image
            
        Returns:
            Cropped PIL Image
        """
        width, height = img.size
        
        crop_size = self._compute_crop_params(width, height, self.resize_target, self.crop_ratio)
        
        crop_transform = CenterCrop(size=crop_size)
        
        return crop_transform(img)

class Augmentation:
    """
    A class for applying various transformations to images.
    """
    
    def __init__(self, rotation=None, blur=None, jitter=None, crop=None, perspective=None):
        """
        Initialize the Augmentation class with transformation(s).
        
        Args:
            rotation: Dict with 'degrees' key. If None or degrees is None, random values will be chosen.
            blur: Dict with 'kernel_size' (default: 7) and 'sigma' (default: 1.5) keys.
            jitter: Dict with 'brightness' (default: 0.2), 'contrast' (default: 0.8), 
                   'saturation' (default: 0.2), 'hue' (default: 0.1) keys.
            crop: Dict with 'resize_target' (default: 256) and 'crop_ratio' (default: 0.80) keys.
            perspective: Dict with 'distortion_scale' (default: 0.3) key.
        """
        rotation = rotation or {}
        blur = blur or {}
        jitter = jitter or {}
        crop = crop or {}
        perspective = perspective or {}
        
        self.rotation = self._init_optional(
            v2.RandomRotation,
            degrees=rotation.get('degrees'),
        )

        self.constrast = self._init_optional(
            v2.ColorJitter,
            brightness=jitter.get('brightness', 0.2),
            contrast=jitter.get('contrast', 0.8),
            saturation=jitter.get('saturation', 0.2),
            hue=jitter.get('hue', 0.1)
        )

        self.blur = self._init_optional(
            v2.GaussianBlur,
            kernel_size=blur.get('kernel_size', 7),
            sigma=blur.get('sigma', 1.5)
        )

        self.crop = self._init_optional(
            Crop,
            resize_target=crop.get('resize_target', 256),
            crop_ratio=crop.get('crop_ratio', 0.80)
        )

        self.perspective = self._init_optional(
            v2.RandomPerspective,
            distortion_scale=perspective.get('distortion_scale', 0.3)
        )

        self.flash = v2.RandomApply([
            v2.ColorJitter(
                brightness=(1.2, 2.0),
                contrast=(0.8, 1.0),
                saturation=(0.7, 1.0)
            )
        ], p=0.5)

        self.transformations = {
            'Rotation': self.rotation,
            'Blur': self.blur,
            'Contrast':  self.constrast,
            'Scaling': self.crop,
            'Illumination': self.flash,
            'Perspective': self.perspective,
        }

    @staticmethod
    def _init_optional(transform_class, **kwargs):
        """Initialize a transform class, only passing non-None keyword arguments."""
        try:
            not_null_filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return transform_class(**not_null_filtered_kwargs)
        except TypeError as e:
            logging.error(f"TypeError in {transform_class.__name__}: {e}. Using default parameters.")
            return transform_class()
        except ValueError as e:
            transform_name = transform_class.__name__
            params_str = ", ".join(f"{k}={v}" for k, v in not_null_filtered_kwargs.items())
            logging.warning(
                f"Invalid parameters for {transform_name}({params_str}): {e}. "
                f"Using default parameters instead."
            )
            return transform_class()
    
    def _load_image(self, img):
        """Load image from path or return PIL Image."""
        if isinstance(img, (str, Path)):
            return Image.open(img)
        return img
    
    def _apply_transform(self, img, transform):
        """Apply a single transformation."""
        img = self._load_image(img)
        return transform(img)
    
    def apply(self, images, transform_name):
        """
        Apply a specific transformation to images.
        
        Args:
            images: A single PIL Image, a list of PIL Images, or a list of image paths
            transform_name: Name of the transformation to apply
            
        Returns:
            If a single image is provided, returns a single transformed image.
            If a list is provided, returns a list of transformed images.
        """
        if transform_name not in self.transformations:
            raise ValueError(f"Unknown transformation: {transform_name}")
        
        transform = self.transformations[transform_name]
        
        if isinstance(images, Image.Image):
            return self._apply_transform(images, transform)
        
        if isinstance(images, (list, tuple)):
            return [self._apply_transform(img, transform) for img in images]
        
        raise TypeError("images must be a PIL Image, a list of PIL Images, or a list of image paths")
    
    def visualize_all_transformations(self, image, figsize=(20, 3)):
        """
        Apply all transformations to an image and plot them horizontally.
        
        Args:
            image: A PIL Image or image path
            figsize: Tuple specifying figure size (width, height)
        """
        orig_img = self._load_image(image)
        
        transformed_images = [orig_img]
        titles = ['Original']
        
        for name, transform in self.transformations.items():
            transformed_img = self._apply_transform(orig_img, transform)
            transformed_images.append(transformed_img)
            titles.append(name)
        
        num_images = len(transformed_images)
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        
        # Handle single image case
        if num_images == 1:
            axes = [axes]
        
        for idx, (img, title) in enumerate(zip(transformed_images, titles)):
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(title, fontsize=10)
        
        plt.tight_layout()
        plt.show()

def plot_images(images, titles=None, figsize=(15, 3)):
    """
    Utility function to plot a list of images in a grid.
    
    Args:
        images: List of image paths (str) or PIL Image objects
        titles: Optional list of titles for each image
        figsize: Tuple specifying figure size (width, height)
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    for idx, img in enumerate(images):
        # Load image if it's a path string or Path object
        if isinstance(img, (str, Path)):
            img = Image.open(img)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
    
    plt.tight_layout()
    plt.show()


# Get the project root directory (two levels up from this script)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
orig_img_path = project_root / "data" / "Apple_Black_rot" / "image (1).JPG"

# Create augmentation instance and visualize all transformations
# All parameters use defaults if not specified
augmentation = Augmentation(
    rotation={'degrees': (15, 345)}  # Only specify what you want to customize
)
augmentation.visualize_all_transformations(orig_img_path)
# img : PIL = augmentation._load_image(orig_img_path)
# print(type(img))
# # print(augmentation._compute_crop_params(img.width, img.height))