from pathlib import Path
from random import Random

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, TensorDataset

# class ImagesLoader(torch.utils.data.DataLoader):

#     def __init__(
#         self,
#         dataset: TensorDataset,
#         original_sizes: list[tuple[int, int]],
#         metas: list[dict],
#         image_set_names: list[str],
#         batch_size=1,
#         shuffle=None,
#         sampler=None,
#         batch_sampler=None,
#         num_workers=0,
#         collate_fn=None,
#         pin_memory=False,
#         drop_last=False,
#         timeout=0,
#         worker_init_fn=None,
#         multiprocessing_context=None,
#         generator=None,
#         *,
#         prefetch_factor=None,
#         persistent_workers=False,
#         pin_memory_device="",
#         in_order=True,
#     ):
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             sampler,
#             batch_sampler,
#             num_workers,
#             collate_fn,
#             pin_memory,
#             drop_last,
#             timeout,
#             worker_init_fn,
#             multiprocessing_context,
#             generator,
#             prefetch_factor=prefetch_factor,
#             persistent_workers=persistent_workers,
#             pin_memory_device=pin_memory_device,
#             in_order=in_order,
#         )

#         self.original_sizes = original_sizes
#         self.metas = metas
#         self.image_set_names = image_set_names

#     def get_original_size(self, index: int):
#         return self.original_sizes[index]

#     def get_meta(self, index: int):
#         return self.metas[index]

#     def get_image_set_name(self, index: int):
#         return self.image_set_names[index]


class TrainValLoaders:

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


class ImagesLoader:

    def __init__(self, image_shape: tuple[int, int], nodata: float):
        self.image_shape = image_shape
        self.nodata = nodata

        self.images = []
        self.metas = []
        self.masks = []
        self.file_names = []
        # self.indices = []
        # self.index_to_position = {}
        self.images_set_names = []
        self.all_images_set_names = []

    def load_data(self, data_folder: Path, images_path: Path, masks_path: Path):
        all_images_paths = sorted(list(images_path.glob("*.tif")))

        Random().shuffle(all_images_paths)

        all_masks_paths = [masks_path / path.name for path in all_images_paths]

        self.all_images_set_names.append(data_folder.name)

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
            # meta["width"] = self.image_shape[1]
            # meta["height"] = self.image_shape[0]

            self.images.append(image_torch)
            self.metas.append(meta)
            self.masks.append(mask_torch)
            self.file_names.append(image_path.name)
            # self.indices.append(len(self.images) - 1)
            # self.index_to_position[self.indices[-1]] = len(self.images) - 1
            self.images_set_names.append(data_folder.name)

    def get_images(self):
        return torch.stack(self.images).unsqueeze(-1).transpose(1, -1)

    def get_masks(self):
        return torch.stack(self.masks).unsqueeze(-1).transpose(1, -1)

    def get_dataloaders(
        self, batch_size: int, train_proportion: float
    ) -> TrainValLoaders:
        print(f"Number of images: {len(self.images)}")
        train_size = int(train_proportion * len(self.images))
        images = self.get_images()
        masks = self.get_masks()
        # indices = torch.tensor(self.indices)

        # Select a random subset of indices
        permuted_indices = torch.randperm(len(self.images))
        train_indices = permuted_indices[:train_size]
        eval_indices = permuted_indices[train_size:]

        dataloader_train = DataLoader(
            TensorDataset(
                images[train_indices],
                masks[train_indices],
                train_indices,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        dataloader_val = DataLoader(
            TensorDataset(
                images[eval_indices],
                masks[eval_indices],
                eval_indices,
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        train_val_loaders = TrainValLoaders(
            train_dataloader=dataloader_train, val_dataloader=dataloader_val
        )

        return train_val_loaders

    def get_file_name(self, index: int):
        return self.file_names[index]

    def get_meta(self, index: int) -> dict:
        return self.metas[index]

    def get_original_size(self, index: int) -> tuple[int, int]:
        return (
            self.metas[index]["width"],
            self.metas[index]["height"],
        )

    def get_image_set_name(self, index: int) -> str:
        return self.images_set_names[index]

    def get_all_image_set_names(self) -> list[str]:
        return self.all_images_set_names
