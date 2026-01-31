import logging
import argparse
import os
from typing import Dict, Optional, List, Tuple, Union, Any
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2, CenterCrop

class Crop(CenterCrop):
    def __init__(self, resize_target: int = 256, crop_ratio: float = 0.6) -> None:
        """
        Initialize Crop transform with parameters for computing crop size.
        
        Args:
            resize_target: Target size for the smaller dimension after resize (default: 256)
            crop_ratio: Ratio of crop size to resize target (default: 0.80)
        """

        super().__init__(size=1)
        self.resize_target: int = resize_target
        self.crop_ratio: float = crop_ratio
    
    @staticmethod
    def _compute_crop_params(
        resize_target: int = 256, 
        crop_ratio: float = 0.80
    ) -> int:
        """
        Compute resize and crop parameters based on image dimensions.
        
        Args:
            resize_target: Target size for the smaller dimension after resize (default: 256)
            crop_ratio: Ratio of crop size to resize target (default: 0.80)
            
        Returns:
            crop_size: Size for CenterCrop transform (int or tuple)
        """
        crop_size: int = int(resize_target * crop_ratio)
        return crop_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Override __call__ to compute crop size dynamically from image dimensions.
        
        Args:
            img: PIL Image
            
        Returns:
            Cropped PIL Image
        """
        width, height = img.size
        
        crop_size: int = self._compute_crop_params(self.resize_target, self.crop_ratio)
        
        crop_transform: CenterCrop = CenterCrop(size=crop_size)
        
        return crop_transform(img)

class Augmentation:
    """
    A class for applying various transformations to images.
    """
    
    def __init__(
        self, 
        rotation: Optional[Dict[str, Any]] = None, 
        blur: Optional[Dict[str, Any]] = None, 
        jitter: Optional[Dict[str, Any]] = None, 
        crop: Optional[Dict[str, Any]] = None, 
        perspective: Optional[Dict[str, Any]] = None,
        flash: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the Augmentation class with transformation(s).
        
        Args:
            rotation: Dict with 'degrees' key. If None or 'degrees' not provided, defaults to (-180, 180).
            blur: Dict with 'kernel_size' (default: 7) and 'sigma' (default: 1.5) keys.
            jitter: Dict with 'brightness' (default: 0.2), 'contrast' (default: 0.8), 
                   'saturation' (default: 0.2), 'hue' (default: 0.1) keys.
            crop: Dict with 'resize_target' (default: 256) and 'crop_ratio' (default: 0.80) keys.
            perspective: Dict with 'distortion_scale' (default: 0.3) key.
            flash: Dict with 'brightness' (default: (2.0, 2.0)), 'contrast' (default: (0.8, 1.0)),
                  'saturation' (default: (0.7, 1.0)), and 'p' (default: 0.5) keys.
        """
        rotation = rotation or {}
        blur = blur or {}
        jitter = jitter or {}
        crop = crop or {}
        perspective = perspective or {}
        flash = flash or {}

        self.rotation: Any = self._init_optional(
            v2.RandomRotation,
            degrees=rotation.get('degrees', (-180, 180)),
        )

        self.constrast: Any = self._init_optional(
            v2.ColorJitter,
            brightness=jitter.get('brightness', 0.2),
            contrast=jitter.get('contrast', 0.8),
            saturation=jitter.get('saturation', 0.2),
            hue=jitter.get('hue', 0.1)
        )

        self.blur: Any = self._init_optional(
            v2.GaussianBlur,
            kernel_size=blur.get('kernel_size', 7),
            sigma=blur.get('sigma', 1.5)
        )

        self.crop: Crop = self._init_optional(
            Crop,
            resize_target=crop.get('resize_target', 256),
            crop_ratio=crop.get('crop_ratio', 0.6)
        )

        self.perspective: Any = self._init_optional(
            v2.RandomPerspective,
            distortion_scale=perspective.get('distortion_scale', 0.3)
        )

        flash_color_jitter: Any = self._init_optional(
            v2.ColorJitter,
            brightness=flash.get('brightness', (2.0, 2.0)),
            contrast=flash.get('contrast', (0.8, 1.0)),
            saturation=flash.get('saturation', (0.7, 1.0))
        )

        self.flash: Any = self._init_optional(
            v2.RandomApply,
            transforms=[flash_color_jitter],
            p=flash.get('p', 0.5)
        )

        self.transformations: Dict[str, Any] = {
            'Rotation': self.rotation,
            'Blur': self.blur,
            'Contrast':  self.constrast,
            'Scaling': self.crop,
            'Illumination': self.flash,
            'Perspective': self.perspective,
        }

    # Instance methods
    def _load_image(self, img: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(img, (str, Path)):
            return Image.open(img)
        return img
    
    def _apply_transform(self, img: Union[str, Path, Image.Image], transform: Any) -> Image.Image:
        """Apply a single transformation."""
        img = self._load_image(img)
        return transform(img)
    
    def apply(
        self, 
        images: Union[Image.Image, List[Image.Image], List[Union[str, Path]]], 
        transform_name: str
    ) -> Union[Image.Image, List[Image.Image]]:
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
    
    def _save_augmented_image(
        self, 
        image: Image.Image, 
        original_path: Union[str, Path], 
        transform_name: str, 
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save an augmented image with the naming convention: original_name_transformname.ext
        
        Args:
            image: PIL Image to save
            original_path: Original image path
            transform_name: Name of the transformation applied
            output_dir: Optional output directory. If None, saves next to original image.
            
        Returns:
            Path to saved image
        """
        original_path = Path(original_path)
        stem = original_path.stem
        suffix = original_path.suffix
        
        new_filename = f"{stem}_{transform_name}{suffix}"
        
        if output_dir:
            output_path = Path(output_dir) / original_path.parent.relative_to(original_path.parents[-1]) / new_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = original_path.parent / new_filename
        
        image.save(output_path)
        return output_path
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Process a single image: apply all transformations and save them.
        
        Args:
            image_path: Path to the image file
            output_dir: Optional output directory. If None, saves next to original image.
            
        Returns:
            List of paths to saved augmented images
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"Image does not exist: {image_path}")
        
        saved_paths: List[Path] = []
        
        for transform_name, transform in self.transformations.items():
            try:
                orig_img: Image.Image = self._load_image(image_path)
                augmented_img: Image.Image = transform(orig_img)
                saved_path: Path = self._save_augmented_image(
                    augmented_img, 
                    image_path, 
                    transform_name,
                    output_dir
                )
                saved_paths.append(saved_path)
            except Exception as e:
                logging.warning(f"Failed to apply {transform_name} to {image_path}: {e}")
        
        return saved_paths
    
    def process_folder(
        self, 
        folder_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, List[Path]]:
        """
        Process all images in a folder recursively.
        
        Args:
            folder_path: Path to the folder containing images
            output_dir: Optional output directory. If None, saves next to original images.
            
        Returns:
            Dictionary mapping original image paths to lists of saved augmented image paths
        """
        image_paths: List[Path] = self._find_images_recursive(folder_path)
        results: Dict[str, List[Path]] = {}
        
        for img_path in image_paths:
            try:
                saved_paths: List[Path] = self.process_image(img_path, output_dir)
                results[str(img_path)] = saved_paths
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")
        
        return results

    # Static methods
    @staticmethod
    def _init_optional(transform_class: type, **kwargs: Any) -> Any:
        """Initialize a transform class, only passing non-None keyword arguments."""
        try:
            not_null_filtered_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
            return transform_class(**not_null_filtered_kwargs)
        except TypeError as e:
            logging.error(f"TypeError in {transform_class.__name__}: {e}. Using default parameters.")
            return transform_class()
        except ValueError as e:
            transform_name: str = transform_class.__name__
            params_str: str = ", ".join(f"{k}={v}" for k, v in not_null_filtered_kwargs.items())
            logging.warning(
                f"Invalid parameters for {transform_name}({params_str}): {e}. "
                f"Using default parameters instead."
            )
            return transform_class()
    
    @staticmethod
    def _find_images_recursive(folder_path: Union[str, Path]) -> List[Path]:
        """
        Recursively find all image files in a folder and its subdirectories.
        
        Args:
            folder_path: Path to the folder to search
            
        Returns:
            List of image file paths
        """
        folder: Path = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        image_extensions: set = {'.JPG'}
        image_paths: List[Path] = []
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                if Path(file).suffix in image_extensions:
                    image_paths.append(Path(root) / file)
        
        return image_paths


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Apply augmentations to images or folders recursively'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to an image file or folder containing images'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for augmented images (default: same as input)'
    )
    
    args: argparse.Namespace = parser.parse_args()

    augmentation: Augmentation = Augmentation()
    
    input_path: Path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return
    
    if input_path.is_file():
        print(f"Processing image: {input_path}")
        saved_paths: List[Path] = augmentation.process_image(input_path, args.output)
        print(f"Saved {len(saved_paths)} augmented images")
        for path in saved_paths:
            print(f"  - {path}")
    elif input_path.is_dir():
        print(f"Processing folder recursively: {input_path}")
        results: Dict[str, List[Path]] = augmentation.process_folder(input_path, args.output)
        total_images: int = len(results)
        total_augmented: int = sum(len(paths) for paths in results.values())
        print(f"Processed {total_images} images")
        print(f"Generated {total_augmented} augmented images")
    else:
        print(f"Error: Invalid path: {input_path}")


if __name__ == '__main__':
    main()