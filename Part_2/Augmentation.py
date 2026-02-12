import logging
import argparse
import shutil
import torch
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import groupby, chain
from typing import Dict, Optional, List, Union, Any
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from plotly.subplots import make_subplots


def plot_augmentations_grid(
    original_img: Image.Image,
    augmented_images: Dict[str, Image.Image],
    cols: int = 3,
):
    """
    Plot original image and augmentations in a grid.

    Args:
        original_img: Original PIL Image
        augmented_images: Dict mapping transform names to augmented images
        cols: Number of columns in the grid
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


class Crop(v2.Transform):
    def __init__(
        self, resize_target: int = 256, crop_ratio: float = 0.8
    ) -> None:
        """
        Args:
            resize_target: Target size for smaller dimension after resize
            crop_ratio: Ratio of crop size to resize target
        """
        super().__init__()
        self.resize_target: int = resize_target
        self.crop_ratio: float = crop_ratio

    @staticmethod
    def _compute_crop_params(
        resize_target: int = 256,
        crop_ratio: float = 0.8
    ) -> int:
        """
        Args:
            resize_target: Target size for smaller dimension after resize
            crop_ratio: Ratio of crop size to resize target

        Returns:
            Crop size for CenterCrop transform
        """
        crop_size: int = int(resize_target * crop_ratio)
        return crop_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor image of shape (C, H, W)

        Returns:
            Cropped tensor image
        """
        crop_size: int = self._compute_crop_params(
            self.resize_target, self.crop_ratio
        )

        crop_transform: v2.CenterCrop = v2.CenterCrop(size=crop_size)
        return crop_transform(img)


class Augmentation:
    """Apply transformations to images."""

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
        Args:
            rotation: Dict with 'degrees' key
            blur: Dict with 'kernel_size' and 'sigma' keys
            jitter: Dict with 'brightness', 'contrast', 'saturation',
                'hue' keys
            crop: Dict with 'resize_target' and 'crop_ratio' keys
            perspective: Dict with 'distortion_scale' key
            flash: Dict with 'brightness', 'contrast', 'saturation' keys
        """
        rotation = rotation or {}
        blur = blur or {}
        jitter = jitter or {}
        crop = crop or {}
        perspective = perspective or {}
        flash = flash or {}

        self.rotation = self._init_optional(
            v2.RandomRotation,
            degrees=rotation.get('degrees', (-180, 180)),
        )

        self.constrast = self._init_optional(
            v2.ColorJitter,
            brightness=jitter.get('brightness', (1.2, 1.2)),
            contrast=jitter.get('contrast', (1.8, 1.8)),
            saturation=jitter.get('saturation', (1.2, 1.2)),
            hue=jitter.get('hue', (0.1, 0.1))
        )

        self.blur = self._init_optional(
            v2.GaussianBlur,
            kernel_size=blur.get('kernel_size', 7),
            sigma=blur.get('sigma', 1.5)
        )

        self.crop: Crop = self._init_optional(
            Crop,
            resize_target=crop.get('resize_target', 256),
            crop_ratio=crop.get('crop_ratio', 0.6)
        )

        self.perspective = self._init_optional(
            v2.RandomPerspective,
            distortion_scale=perspective.get('distortion_scale', 0.5),
            p=1.0
        )

        self.flash = self._init_optional(
            v2.ColorJitter,
            brightness=flash.get('brightness', (2.0, 2.0)),
            contrast=flash.get('contrast', (0.8, 1.0)),
            saturation=flash.get('saturation', (0.7, 1.0))
        )

        self._to_image = v2.ToImage()
        self._to_dtype = v2.ToDtype(torch.float32, scale=True)

        self.transformations: Dict[str, Any] = {
            'Rotation': self.rotation,
            'Blur': self.blur,
            'Contrast': self.constrast,
            'Scaling': self.crop,
            'Illumination': self.flash,
            'Perspective': self.perspective,
        }

    # Static methods
    @staticmethod
    def _init_optional(transform_class: type, **kwargs: Any) -> Any:
        """
        Initialize a transform class, only passing non-None keyword arguments.

        Args:
            transform_class: Transform class to instantiate
            **kwargs: Keyword arguments for the transform class

        Returns:
            Instantiated transform object
        """
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

    # Instance methods
    def _load_image(self, img: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Args:
            img: Image path or PIL Image

        Returns:
            PIL Image
        """
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

    def _save_augmented_image(
        self,
        image: Image.Image,
        original_path: Union[str, Path],
        transform_name: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Args:
            image: Augmented PIL Image to save
            original_path: Path to the original image
            transform_name: Name of the transformation applied
            output_dir: Directory where the augmented image should be saved
        """
        try:
            original_path = Path(original_path)
            base_name = original_path.stem
            
            if output_dir:
                transformed_image_path = (
                    f'{output_dir}/{base_name}_{transform_name}.JPG'
                )
            else:
                transformed_image_path = (
                    f'{base_name}_{transform_name}.JPG'
                )
            image.save(transformed_image_path)
        except Exception as e:
            print(f'Error saving augmented image: {e}')

    def apply_all_transformations(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Image.Image]:
        """
        Apply all transformations to an image and save them.

        Args:
            image_path: Path to the input image
            output_dir: Directory where augmented images should be saved
        """
        orig_pil = self._load_image(image_path)

        saved_paths: Dict[str, Image.Image] = {}

        for transform_name, transform in self.transformations.items():
            try:
                augmented_img = transform(orig_pil)
                self._save_augmented_image(
                    augmented_img,
                    image_path,
                    transform_name,
                    output_dir
                )
                saved_paths[transform_name] = augmented_img
            except Exception as e:
                logging.warning(
                    f"Failed to apply {transform_name} to {image_path}: {e}"
                )
        return saved_paths

    def _copy_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> None:
        """
        Copy all images preserving class directory structure.

        Args:
            dataset_path: Source dataset path
            output_dir: Destination directory
        """

        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)

        original_images = list(dataset_path.rglob("*.JPG"))

        for img in tqdm(
            original_images,
            desc=f'Copying images from {dataset_path} to {output_dir}',
        ):
            try:
                class_dir = img.parent.name
                target_dir = output_dir / class_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / img.name
                shutil.copy2(img, target_path)
            except Exception as e:
                logging.warning(f"Failed to copy {img}: {e}")
                continue

    def _get_class_images_and_output_dir(
        self,
        dataset_path: Path,
        output_dir: Path
    ) -> List[Dict[str, Union[Path, List[Path]]]]:

        images = sorted(
            dataset_path.rglob("*.JPG"),
            key=lambda p: p.parent.name
        )

        result = []
        for class_dir, group in groupby(images, key=lambda p: p.parent.name):
            result.append({
                "output_dir": output_dir / class_dir,
                "images": list(group)
            })

        return result

    def augment_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "augmented_directory"
    ) -> None:
        """
        Copy dataset and apply all transformations to each image.

        Args:
            dataset_path: Path to the dataset directory
            output_dir: Output directory for augmented images
        """
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)

        if not dataset_path.exists() or not dataset_path.is_dir():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        self._copy_dataset(dataset_path, output_dir)

        class_groups = self._get_class_images_and_output_dir(
            dataset_path, output_dir
        )

        all_images = list(chain.from_iterable(
            group["images"] for group in class_groups
        ))

        with tqdm(total=len(all_images), desc="Augmenting dataset") as pbar:
            for class_group in class_groups:
                output_category_dir = class_group["output_dir"]
                images = class_group["images"]

                for image_path in images:
                    try:
                        self.apply_all_transformations(
                            image_path,
                            output_category_dir
                        )
                    except Exception as e:
                        logging.warning(f"Failed to augment {image_path}: {e}")
                    finally:
                        pbar.update(1)


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
        print(f"Applying transformations to {input_path}...")
        try:
            augmented_images = augmentation.apply_all_transformations(
                input_path
            )
            original_img = augmentation._load_image(input_path)
            plot_augmentations_grid(original_img, augmented_images)
        except Exception as e:
            logging.warning(f"Failed to augmented single image: {e}")

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
