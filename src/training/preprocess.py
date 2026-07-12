import os
import random

from PIL import (
    Image,
    ImageOps,
)

from src.utils.logger import training_logger as logger


class DatasetProcessor:
    def __init__(
        self,
        input_dir: str,
        img_size: tuple[int, int] = (640, 640),
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.input_dir = input_dir
        self.img_size = img_size
        self.images = {}
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        random.seed(self.seed)

    def load_images(self):
        """
        Load images from the input directory and store them in the images list.
        """
        logger.info(f"Loading images from {self.input_dir}...")
        processed_folders, failed_folders = 0, 0
        processed_images, failed_images = 0, 0

        for classe in sorted(os.listdir(self.input_dir)):
            class_folder = os.path.join(self.input_dir, classe)

            if not os.path.isdir(class_folder):
                failed_folders += 1
                logger.warning(
                    f"[PROCESSING] Skipping non-directory file: {class_folder}"
                )
                continue

            logger.info(f"[LOADING] Processing class: {classe}")

            # Collect all image files in the class folder
            images = []
            for image_index, image_name in enumerate(sorted(os.listdir(class_folder))):
                logger.info(
                    f"[LOADING] Processing image {image_index + 1}/{len(os.listdir(class_folder))}"
                )

                if not image_name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp", ".avif")
                ):
                    failed_images += 1
                    logger.warning(f"[LOADING] Skipping non-image file: {image_name}")
                    continue

                images.append(image_name)
                processed_images += 1

            if not images:
                failed_folders += 1
                logger.warning(
                    f"[LOADING] No valid images found in class folder: {class_folder}"
                )
                continue

            self.images[classe] = images
            processed_folders += 1

        logger.info(f"[LOADING] Gathered images for {len(self.images)} classes.")
        logger.info(
            f"[LOADING] Finished loading images. Processed folders: {processed_folders}, Failed folders: {failed_folders}, Processed images: {processed_images}, Failed images: {failed_images}"
        )

    def split_dataset(self):
        processed_images, failed_images = 0, 0

        for classe, images in self.images.items():
            random.shuffle(images)
            total_images = len(images)
            train_end = int(total_images * self.train_ratio)
            val_end = train_end + int(total_images * self.val_ratio)

            train_images = images[:train_end]
            val_images = images[train_end:val_end]
            test_images = images[val_end:]

            splits = {"train": train_images, "val": val_images, "test": test_images}

            for split_name, split_images in splits.items():
                logger.info(
                    f"[SPLIT] Class: {classe}, Split: {split_name}, Number of images: {len(split_images)}"
                )
                split_dir = os.path.join("dataset", split_name, classe)
                os.makedirs(split_dir, exist_ok=True)

                for image_index, image_name in enumerate(split_images):
                    image_path = os.path.join(self.input_dir, classe, image_name)

                    try:
                        img = Image.open(image_path).convert("RGB")
                        resized_img = self._resize_img(img)
                        resized_img.save(
                            os.path.join(split_dir, f"{image_index + 1}.jpg"),
                            quality=95,
                        )
                        processed_images += 1

                    except Exception as e:
                        failed_images += 1
                        logger.error(
                            f"[SPLIT] Error processing image {image_path}: {e}"
                        )

        logger.info(
            f"[SPLIT] Finished splitting dataset. Processed images: {processed_images}, Failed images: {failed_images}"
        )

    def _resize_img(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(
            img,
            self.img_size,
            method=Image.Resampling.LANCZOS,
        )
