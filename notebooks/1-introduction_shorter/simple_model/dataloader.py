from pathlib import Path
from random import Random

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, TensorDataset


class ImagesLoader:

    def __init__(self, image_shape: tuple[int, int], nodata: float):
        self.image_shape = image_shape
        self.nodata = nodata

        self.images = []
        self.metas = []
        self.masks = []
        self.file_names = []
        self.indices = []
        self.index_to_position = {}

    def load_data(self, images_path: Path, masks_path: Path):
        all_images_paths = sorted(list(images_path.glob("*.tif")))
        # Set random seed

        Random(42).shuffle(all_images_paths)
        all_masks_paths = [masks_path.joinpath(path.name) for path in all_images_paths]

        for image_path, mask_path in zip(all_images_paths, all_masks_paths):
            with rasterio.open(image_path) as src:
                image = src.read().squeeze()
                image = np.where(image == src.nodata, self.nodata, image)
                meta = src.meta.copy()
            with rasterio.open(mask_path) as src:
                mask = src.read().squeeze()
            image_torch = torch.tensor(image, dtype=torch.float32)
            mask_torch = torch.tensor(mask, dtype=torch.bool)

            if image_torch.shape != self.image_shape:
                image_torch = torch.nn.functional.pad(
                    image_torch,
                    (
                        0,
                        self.image_shape[1] - image_torch.shape[1],
                        0,
                        self.image_shape[0] - image_torch.shape[0],
                    ),
                )
                mask_torch = torch.nn.functional.pad(
                    mask_torch,
                    (
                        0,
                        self.image_shape[1] - mask_torch.shape[1],
                        0,
                        self.image_shape[0] - mask_torch.shape[0],
                    ),
                )
                meta["width"] = self.image_shape[1]
                meta["height"] = self.image_shape[0]

            self.images.append(image_torch)
            self.metas.append(meta)
            self.masks.append(mask_torch)
            self.file_names.append(image_path.name)
            self.indices.append(int(image_path.stem.split("_")[-1]))
            self.index_to_position[self.indices[-1]] = len(self.images) - 1

    def get_images(self):
        return torch.stack(self.images).unsqueeze(-1).transpose(1, -1)

    def get_masks(self):
        return torch.stack(self.masks).unsqueeze(-1).transpose(1, -1)

    def get_dataloaders(self, batch_size: int, train_proportion: float):
        train_size = int(train_proportion * len(self.images))
        images = self.get_images()
        masks = self.get_masks()
        indices = torch.tensor(self.indices)

        self.dataloader_train = DataLoader(
            TensorDataset(
                images[:train_size],
                masks[:train_size],
                indices[:train_size],
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        self.dataloader_val = DataLoader(
            TensorDataset(
                images[train_size:],
                masks[train_size:],
                indices[train_size:],
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        return self.dataloader_train, self.dataloader_val

    def get_file_name(self, index: int):
        return self.file_names[self.index_to_position[index]]

    def get_meta(self, index: int):
        return self.metas[self.index_to_position[index]]


# def load_data(train_proportion: float):
#     image_shape = (512, 512)
#     images_path = Path("data/tiles/image")
#     masks_path = Path("data/tiles/mask")

#     images_paths = list(images_path.glob("*.tif"))
#     masks_paths = [masks_path.joinpath(path.name) for path in images_paths]

#     images = []
#     metas = []
#     masks = []

#     for image_path, mask_path in zip(images_paths, masks_paths):
#         with rasterio.open(image_path) as src:
#             image = src.read().squeeze()
#             # Replace nodata values with 0
#             image = np.where(image == src.nodata, 0, image)
#             # Copy the metadata
#             meta = src.meta.copy()
#         with rasterio.open(mask_path) as src:
#             mask = src.read().squeeze()
#         image_torch = torch.tensor(image, dtype=torch.float32)
#         mask_torch = torch.tensor(mask, dtype=torch.bool)

#         if image_torch.shape != image_shape:
#             image_torch = torch.nn.functional.pad(
#                 image_torch,
#                 (
#                     0,
#                     image_shape[1] - image_torch.shape[1],
#                     0,
#                     image_shape[0] - image_torch.shape[0],
#                 ),
#             )
#             mask_torch = torch.nn.functional.pad(
#                 mask_torch,
#                 (
#                     0,
#                     image_shape[1] - mask_torch.shape[1],
#                     0,
#                     image_shape[0] - mask_torch.shape[0],
#                 ),
#             )

#         images.append(image_torch)
#         metas.append(meta)
#         masks.append(mask_torch)

#     images_torch = torch.stack(images).unsqueeze(-1).transpose(1, -1)
#     masks_torch = torch.stack(masks).unsqueeze(-1).transpose(1, -1)

#     train_size = int(train_proportion * len(images_torch))

#     dataloaders = (
#         DataLoader(
#             TensorDataset(
#                 images_torch[:train_size],
#                 masks_torch[:train_size],
#             ),
#             batch_size=16,
#             shuffle=False,
#         ),
#         DataLoader(
#             TensorDataset(
#                 images_torch[train_size:],
#                 masks_torch[train_size:],
#             ),
#             batch_size=16,
#             shuffle=False,
#         ),
#     )

#     return image_shape, dataloaders, metas
