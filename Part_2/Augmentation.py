import logging
import argparse
import os
from tqdm import tqdm
from typing import Dict, Iterator, Optional, Union, Any
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2, CenterCrop

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_augmentations_grid(
    original_img: Image.Image,
    augmented_images: Dict[str, Image.Image],
    cols: int = 3,
):
    """
    Plot original image and its augmentations in a grid using Plotly.
    """

    all_images = {"Original": original_img, **augmented_images}
    names = list(all_images.keys())
    images = list(all_images.values())

    rows = int(np.ceil(len(images) / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names
    )

    for idx, (name, img) in enumerate(all_images.items()):
        row = idx // cols + 1
        col = idx % cols + 1

        img_np = np.array(img)

        fig.add_trace(
            go.Image(z=img_np),
            row=row,
            col=col
        )

    fig.update_layout(
        height=300 * rows,
        width=300 * cols,
        title_text="Image Augmentations Grid",
        showlegend=False
    )

    fig.show()


class Crop(CenterCrop):
    def __init__(
        self, resize_target: int = 256, crop_ratio: float = 0.6
    ) -> None:
        """
        Initialize Crop transform with parameters for computing crop size.

        Args:
            resize_target: Target size for the smaller dimension after
                resize (default: 256)
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
            resize_target: Target size for the smaller dimension after
                resize (default: 256)
            crop_ratio: Ratio of crop size to resize target (default: 0.80)

        Returns:
            crop_size: Size for CenterCrop transform (int or tuple)
        """
        crop_size: int = int(resize_target * crop_ratio)
        return crop_size

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Override __call__ to compute crop size dynamically from image
        dimensions.

        Args:
            img: PIL Image

        Returns:
            Cropped PIL Image
        """
        width, height = img.size

        crop_size: int = self._compute_crop_params(
            self.resize_target, self.crop_ratio
        )

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
            rotation: Dict with 'degrees' key. If None or 'degrees' not
                provided, defaults to (-180, 180).
            blur: Dict with 'kernel_size' (default: 7) and 'sigma'
                (default: 1.5) keys.
            jitter: Dict with 'brightness' (default: 0.2), 'contrast'
                (default: 0.8), 'saturation' (default: 0.2), 'hue'
                (default: 0.1) keys.
            crop: Dict with 'resize_target' (default: 256) and 'crop_ratio'
                (default: 0.80) keys.
            perspective: Dict with 'distortion_scale' (default: 0.3) key.
            flash: Dict with 'brightness' (default: (2.0, 2.0)),
                'contrast' (default: (0.8, 1.0)), 'saturation'
                (default: (0.7, 1.0)), and 'p' (default: 0.5) keys.
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
            'Contrast': self.constrast,
            'Scaling': self.crop,
            'Illumination': self.flash,
            'Perspective': self.perspective,
        }

    # Instance methods
    def _load_image(self, img: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(img, (str, Path)):
            try:
                return Image.open(img)
            except (
                FileNotFoundError,
                OSError,
                Image.UnidentifiedImageError
            ) as e:
                raise ValueError(
                    f"Cannot load image from {img}: {e}"
                ) from e
        return img

    def _apply_transform(
        self, img: Union[str, Path, Image.Image], transform: Any
    ) -> Image.Image:
        """Apply a single transformation."""
        img = self._load_image(img)
        return transform(img)

    def apply(
        self,
        images: Union[
            Image.Image, List[Image.Image], List[Union[str, Path]]
        ],
        transform_name: str
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Apply a specific transformation to images.

        Args:
            images: A single PIL Image, a list of PIL Images, or a list of
                image paths
            transform_name: Name of the transformation to apply

        Returns:
            If a single image is provided, returns a single transformed
            image. If a list is provided, returns a list of transformed
            images.
        """
        if transform_name not in self.transformations:
            raise ValueError(f"Unknown transformation: {transform_name}")

        transform = self.transformations[transform_name]

        if isinstance(images, Image.Image):
            return self._apply_transform(images, transform)

        if isinstance(images, (list, tuple)):
            return [
                self._apply_transform(img, transform) for img in images
            ]

        raise TypeError(
            "images must be a PIL Image, a list of PIL Images, "
            "or a list of image paths"
        )

    def _save_augmented_image(
        self,
        image: Image.Image,
        original_path: Union[str, Path],
        transform_name: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Compute the output path for an augmented image.

        Args:
            image_path: Path to the original image
            transform_name: Name of the transformation to apply
            output_dir: Directory where the augmented image should be saved

        Returns:
            Path to saved image
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        stem = image_path.stem
        suffix = image_path.suffix
        new_filename = f"{stem}_{transform_name}{suffix}"
        output_path = output_dir / new_filename

        return output_path

    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """
        Apply a transformation to an image and optionally save it.

        Args:
            image_path: Path to the input image
            transform_name: Name of the transformation to apply
            output_path: Optional path to save the augmented image

        Returns:
            Augmented PIL Image if successful, None otherwise
        """
        try:
            orig_img = self._load_image(image_path)

            transform = self.transformations[transform_name]
            augmented_img = transform(orig_img)

            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                augmented_img.save(output_path)

            return augmented_img
        except (ValueError, OSError, KeyError) as e:
            logging.warning(
                f"Failed to apply {transform_name} to {image_path}: {e}"
            )
            return None

    def _apply_all_transformations(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Image.Image]:
        """
        Apply all transformations to an image and save them.

        Args:
            image_path: Path to the input image
            output_dir: Optional directory where augmented images should be
                       saved. If None, uses the directory of the input image.

        Returns:
            Dictionary mapping transform names to augmented PIL Images
        """
        transform_names = list(self.transformations.keys())

        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)

        augmented_images: Dict[str, Image.Image] = {}

        for transform_name in transform_names:
            output_path = self._compute_output_path(
                image_path, transform_name, output_dir
            )

            augmented_img = self._apply_transform_to_image(
                image_path, transform_name, output_path
            )
            if augmented_img is not None:
                augmented_images[transform_name] = augmented_img

        return augmented_images

    def _copy_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        images: Iterator[Path]
    ) -> None:
        """Copy all images to output directory preserving structure."""

        for image in tqdm(
            images,
            desc=f'Copying images from {dataset_path} to {output_dir}',
            total=len(os.listdir(dataset_path))
        ):
            try:
                save_path = Path(
                    f'{output_dir}/',
                    image.parent.stem
                )
                os.makedirs(save_path, exist_ok=True)

                img = self._load_image(image)
                output_path = save_path / image.name
                img.save(output_path)
            except Exception as e:
                logging.warning(f"Failed to copy {image}: {e}")
                continue

    def augment_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "augmented_directory"
    ) -> None:
        """Copy dataset and apply all transformations to each image."""
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)

        if not dataset_path.exists() or not dataset_path.is_dir():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        images = Path(dataset_path).glob('**/*.JPG')

        self._copy_dataset(dataset_path, output_dir, images)

        for subdirectory_class in tqdm(
            Path(dataset_path).iterdir(),
            desc=f'Augmenting images from {dataset_path}',
            total=len(os.listdir(dataset_path))
        ):
            if not subdirectory_class.is_dir():
                continue

            rel_path = subdirectory_class.relative_to(dataset_path)
            output_category_dir = output_dir / rel_path

            images = list(subdirectory_class.glob('**/*.JPG'))

            for image_path in images:
                try:
                    self._apply_all_transformations(
                        image_path,
                        output_category_dir)
                except Exception as e:
                    logging.warning(f"Failed to augment {image_path}: {e}")
                    continue

    # Static methods
    @staticmethod
    def _init_optional(transform_class: type, **kwargs: Any) -> Any:
        """Initialize a transform class, only passing non-None keyword
        arguments."""
        try:
            not_null_filtered_kwargs: Dict[str, Any] = {
                k: v for k, v in kwargs.items() if v is not None
            }
            return transform_class(**not_null_filtered_kwargs)
        except TypeError as e:
            logging.error(
                f"TypeError in {transform_class.__name__}: {e}. "
                "Using default parameters."
            )
            return transform_class()
        except ValueError as e:
            transform_name: str = transform_class.__name__
            params_str: str = ", ".join(
                f"{k}={v}" for k, v in not_null_filtered_kwargs.items()
            )
            logging.warning(
                f"Invalid parameters for {transform_name}({params_str}): {e}. "
                "Using default parameters instead."
            )
            return transform_class()


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
        help='Output directory for augmented images '
             '(default: same as input)'
    )

    args: argparse.Namespace = parser.parse_args()

    augmentation: Augmentation = Augmentation()

    input_path: Path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return

    if input_path.is_file():
        print(f"Applying transformations to {input_path.name}...")
        augmented_images = augmentation._apply_all_transformations(
            input_path
        )
        if augmented_images:
            try:
                original_img = augmentation._load_image(input_path)
                plot_augmentations_grid(original_img, augmented_images)
            except Exception as e:
                logging.warning(f"Failed to plot images: {e}")

    elif input_path.is_dir():
        try:
            augmentation.augment_dataset(
                input_path
            )
        except Exception as e:
            logging.exception(f"Error during balancing: {e}")
            return
    else:
        print(f"Error: Invalid path: {input_path}")


if __name__ == '__main__':
    main()
