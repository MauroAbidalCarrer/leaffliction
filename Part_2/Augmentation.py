import logging
import argparse
import shutil
import torch
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import groupby
from typing import Dict, Optional, List, Union, Any
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from plotly.subplots import make_subplots


@dataclass
class ClassGroup:
    """Class group structure with getter methods."""
    class_name: str
    output_dir: Path
    images: List[Path]

    def get_class_name(self) -> str:
        """Get class name."""
        return self.class_name

    def get_output_dir(self) -> Path:
        """Get output directory."""
        return self.output_dir

    def get_images(self) -> List[Path]:
        """Get images list."""
        return self.images


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


def _group_images_by_class(dataset_path: Path) -> Dict[str, List[Path]]:
    """
    Group images by their class directory.

    Args:
        dataset_path: Path to the source dataset directory

    Returns:
        Dict mapping class names to lists of image paths
    """
    images = sorted(
        dataset_path.rglob("*.JPG"),
        key=lambda p: p.parent.name
    )
    grouped = {}
    for class_dir, group in groupby(images, key=lambda p: p.parent.name):
        grouped[class_dir] = list(group)
    return grouped


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
        except (TypeError, ValueError) as e:
            logging.warning(f"{transform_class.__name__}: {e}, using defaults")
            return transform_class()

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


class Balance:
    """Balance dataset classes using augmentation."""

    def __init__(self, augmentation: Augmentation) -> None:
        """
        Args:
            augmentation: Augmentation instance to use for creating
                augmented images
        """
        self.augmentation = augmentation

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

    def _count_images_per_class(
        self,
        dataset_path: Path
    ) -> Dict[str, int]:
        """
        Count original images per class in the source dataset.

        Args:
            dataset_path: Path to the source dataset directory

        Returns:
            Dict mapping class names to image counts
        """
        images = sorted(
            dataset_path.rglob("*.JPG"),
            key=lambda p: p.parent.name
        )
        class_counts: Dict[str, int] = {}
        for class_dir, group in groupby(images, key=lambda p: p.parent.name):
            class_counts[class_dir] = len(list(group))
        return class_counts

    def _get_current_count(self, output_dir: Path) -> int:
        """
        Get current image count in a directory.

        Args:
            output_dir: Directory to count images in

        Returns:
            Number of JPG images in the directory
        """
        return len(list(output_dir.glob("*.JPG")))

    def _get_class_images_and_output_dir(
        self,
        dataset_path: Path,
        output_dir: Path
    ) -> List[ClassGroup]:
        """
        Get images grouped by class with their output directories.

        Args:
            dataset_path: Path to the source dataset
            output_dir: Path to the output directory

        Returns:
            List of ClassGroup instances
        """
        images = sorted(
            dataset_path.rglob("*.JPG"),
            key=lambda p: p.parent.name
        )

        result: List[ClassGroup] = []
        for class_name, group in groupby(images, key=lambda p: p.parent.name):
            result.append(ClassGroup(
                class_name=class_name,
                output_dir=output_dir / class_name,
                images=list(group)
            ))

        return result

    def _update_progress(
        self,
        output_dir: Path,
        current_count: int,
        target_count: int,
        pbar: tqdm
    ) -> int:
        """
        Calculate progress and update progress bar.

        Args:
            output_dir: Directory to count images in
            current_count: Current image count
            target_count: Target image count
            pbar: Progress bar to update

        Returns:
            New current count after update
        """
        new_count = self._get_current_count(output_dir)
        images_created = new_count - current_count
        remaining_needed = target_count - current_count
        progress_update = min(images_created, remaining_needed)
        pbar.update(progress_update)
        return new_count

    def _augment_class_until_target(
        self,
        class_group: ClassGroup,
        target_count: int,
        initial_count: int,
        pbar: tqdm
    ) -> None:
        """
        Augment images for a class until target count is reached.

        Args:
            class_group: ClassGroup containing class information
            target_count: Target number of images to reach
            initial_count: Current count of images in output directory
            pbar: Progress bar to update
        """
        images = class_group.get_images()
        output_dir = class_group.get_output_dir()
        current_count = initial_count
        image_idx = 0

        while current_count < target_count:
            if image_idx >= len(images):
                image_idx = 0

            image_path = images[image_idx]

            try:
                self.augmentation.apply_all_transformations(
                    image_path,
                    output_dir
                )

                current_count = self._update_progress(
                    output_dir,
                    current_count,
                    target_count,
                    pbar
                )

                if current_count >= target_count:
                    break

            except Exception as e:
                logging.warning(f"Failed to augment {image_path}: {e}")

            image_idx += 1

    def _validate_and_get_target_count(
        self,
        class_counts: Dict[str, int]
    ) -> int:
        """
        Validate dataset and calculate target count for balancing.

        Args:
            class_counts: Dict mapping class names to image counts

        Returns:
            Target count (maximum class count)

        Raises:
            ValueError: If dataset is invalid or classes have insufficient
                source images
        """
        if not class_counts or len(class_counts.keys()) <= 1:
            raise ValueError("Dataset must have at least 2 classes")

        target_count = max(class_counts.values())
        num_transforms = len(self.augmentation.transformations)

        for class_name, count in class_counts.items():
            needed = target_count - count
            if needed > 0:
                images_per_cycle = count * num_transforms
                if images_per_cycle == 0 or needed > images_per_cycle * 100:
                    raise ValueError(
                        f"Class {class_name} has insufficient source images. "
                    )

        return target_count

    def balance_dataset(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "augmented_directory"
    ) -> None:
        """
        Copy dataset and apply transformations to balance classes to
        target count.

        Args:
            dataset_path: Path to the dataset directory
            output_dir: Output directory for augmented images
        """
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)

        if not dataset_path.exists() or not dataset_path.is_dir():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        class_counts = self._count_images_per_class(dataset_path)

        target_count = self._validate_and_get_target_count(class_counts)

        self._copy_dataset(dataset_path, output_dir)

        class_groups = self._get_class_images_and_output_dir(
            dataset_path, output_dir
        )

        total_needed = sum(
            max(0, target_count - class_counts.get(group.get_class_name(), 0))
            for group in class_groups
        )

        with tqdm(total=total_needed, desc="Balancing dataset") as pbar:
            for class_group in class_groups:
                class_name = class_group.get_class_name()
                initial_count = class_counts.get(class_name, 0)
                needed = target_count - initial_count

                if needed <= 0:
                    continue

                self._augment_class_until_target(
                    class_group,
                    target_count,
                    initial_count,
                    pbar
                )


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
            balance = Balance(augmentation)
            balance.balance_dataset(
                input_path
            )
        except Exception as e:
            logging.error(f"Error during balancing: {e}")
            return
    else:
        print(f"Error: Invalid path: {input_path}")


if __name__ == '__main__':
    main()
