from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm.autonotebook import tqdm

from simple_model.dataloader import ImagesLoader
from simple_model.dataparse import merge_tiff_files
from simple_model.visualisation import TrainingVisualisation

# # Function to check if running inside a Jupyter Notebook
# def is_running_in_notebook():
#     """Detect whether the script is running inside a Jupyter Notebook."""
#     try:
#         from IPython import get_ipython

#         return get_ipython() is not None
#     except ImportError:
#         return False


# # Import correct tqdm version
# if is_running_in_notebook():
#     from tqdm.notebook import tqdm  # Jupyter-friendly
# else:
#     from tqdm import tqdm  # Standard terminal tqdm


def dice_loss(logits, targets, reduction: str, smooth: float = 1.0):
    preds = torch.sigmoid(logits)
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    loss = 1 - ((2.0 * intersection + smooth) / (union + smooth))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def full_loss(logits, targets):
    alpha = 1
    beta = 1
    gamma = 1
    sum_weights = alpha + beta + gamma
    alpha /= sum_weights
    beta /= sum_weights
    gamma /= sum_weights

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    dice = dice_loss(logits, targets, reduction="mean")
    focal = sigmoid_focal_loss(logits, targets, reduction="mean")

    total = alpha * bce + beta * dice + gamma * focal
    return total


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_reduced = F.max_pool2d(x, kernel_size=2)

        return x, x_reduced


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            2 * in_channels, in_channels, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class SegmentationConvolutionalNetwork(nn.Module):

    def __init__(self, image_size: tuple[int, int], input_channels: int = 1):
        super().__init__()

        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        encoder_steps = [input_channels, 16, 32, 64]
        self.encoders = nn.ModuleList(
            [
                Downsample(encoder_steps[i], encoder_steps[i + 1])
                for i in range(len(encoder_steps) - 1)
            ]
        )

        # Decoder
        decoder_steps = encoder_steps[:]
        decoder_steps[0] = 1
        self.decoders = nn.ModuleList(
            [
                Upsample(decoder_steps[i + 1], decoder_steps[i])
                for i in range(len(encoder_steps) - 2, -1, -1)
            ]
        )

        self.final_conv = nn.Conv2d(decoder_steps[0], 1, kernel_size=1)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_outputs = []
        x_down = x
        for encoder in self.encoders:
            x, x_down = encoder(x_down)
            encoder_outputs.append(x)

        # Decoder
        x = x_down
        for decoder, e in zip(self.decoders, encoder_outputs[::-1]):
            x = decoder(x, e)

        # Final convolution
        d0 = self.final_conv(x)

        return d0

    def training_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    ) -> float:
        logits = self.forward(images)
        loss = full_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)

        return loss.nanmean().item()

    def validation_step(self, images: torch.Tensor, targets: torch.Tensor) -> float:
        logits = self.forward(images)
        loss = full_loss(logits, targets)

        return loss.nanmean().item()

    def run(
        self,
        dataloaders: tuple[torch.utils.data.DataLoader],
        epochs: int,
        stop_early_after: int = 10,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        scheduler = None
        progress = tqdm(range(epochs))

        # Best model and early stopping
        best_loss = float("inf")
        best_model = None
        best_loss_epoch = 0

        # Visualisation
        visualisation = TrainingVisualisation()

        all_mean_losses = dict(training=[], validation=[])

        for epoch in progress:
            losses = dict(training=[], validation=[])

            self.train()

            for images, targets, indices in dataloaders[0]:
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                losses["training"].append(
                    self.training_step(images, targets, optimizer, scheduler)
                )

            with torch.no_grad():
                self.eval()

                for images, targets, indices in dataloaders[1]:
                    images = images.to(self.device)
                    targets = targets.float().to(self.device)
                    losses["validation"].append(self.validation_step(images, targets))

            mean_training_loss = sum(losses["training"]) / len(losses["training"])
            mean_validation_loss = sum(losses["validation"]) / len(losses["validation"])

            progress.set_description(
                "[training.loss] {:.4f}, [validation.loss] {:.4f}".format(
                    mean_training_loss, mean_validation_loss
                ),
                refresh=True,
            )

            all_mean_losses["training"].append(mean_training_loss)
            all_mean_losses["validation"].append(mean_validation_loss)

            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            visualisation.update_plots(
                epoch=epoch,
                train_loss=mean_training_loss,
                val_loss=mean_validation_loss,
                lr=current_lr,
            )

            if mean_validation_loss < best_loss:
                best_loss = mean_validation_loss
                best_model = self.state_dict()
                best_loss_epoch = epoch
            else:
                if epoch - best_loss_epoch > stop_early_after:
                    break

        self.load_state_dict(best_model)

    def save_predictions(
        self,
        images_loader: ImagesLoader,
        dataloaders: tuple[torch.utils.data.DataLoader],
        prediction_folder: Path,
        error_folder: Path,
        probabilities_folder: Path,
    ):
        prediction_folder.mkdir(parents=True, exist_ok=True)
        error_folder.mkdir(parents=True, exist_ok=True)
        probabilities_folder.mkdir(parents=True, exist_ok=True)

        train_folder = "train"
        val_folder = "val"

        self.eval()
        with torch.no_grad():
            for dataloader, folder in zip(dataloaders, [train_folder, val_folder]):
                # Create subfolders for predictions and errors
                current_prediction_folder = prediction_folder.joinpath(folder)
                current_prediction_tiles_folder = current_prediction_folder.joinpath(
                    "tiles"
                )
                current_prediction_tiles_folder.mkdir(parents=True, exist_ok=True)

                current_error_folder = error_folder.joinpath(folder)
                current_error_tiles_folder = current_error_folder.joinpath("tiles")
                current_error_tiles_folder.mkdir(parents=True, exist_ok=True)

                current_probabilities_folder = probabilities_folder.joinpath(folder)
                current_probabilities_tiles_folder = (
                    current_probabilities_folder.joinpath("tiles")
                )
                current_probabilities_tiles_folder.mkdir(parents=True, exist_ok=True)

                # Store files paths for merging
                prediction_files = []
                error_files = []
                probabilities_files = []

                for images, targets, indices in dataloader:
                    images = images.to(self.device)
                    logits = self.forward(images)
                    probabilities = torch.sigmoid(logits)

                    probabilities = probabilities.squeeze().detach().cpu().numpy()
                    targets = targets.squeeze().detach().cpu().numpy()

                    # Transpose to write the image with rasterio
                    probabilities = np.transpose(probabilities, (0, 2, 1))
                    targets = np.transpose(targets, (0, 2, 1))

                    # Set predictions to 0 or 1
                    predictions = np.where(probabilities > 0.5, 1, 0)

                    for probability, prediction, target, index in zip(
                        probabilities, predictions, targets, indices
                    ):
                        index = index.item()
                        image_file_name = images_loader.get_file_name(index)
                        image_meta = images_loader.get_meta(index)

                        # Get the file paths
                        prediction_path = current_prediction_tiles_folder.joinpath(
                            image_file_name
                        )
                        error_path = current_error_tiles_folder.joinpath(
                            image_file_name
                        )
                        probabilities_path = (
                            current_probabilities_tiles_folder.joinpath(image_file_name)
                        )
                        prediction_files.append(prediction_path)
                        error_files.append(error_path)
                        probabilities_files.append(probabilities_path)

                        image_meta.update({"dtype": "uint8", "nodata": 255})
                        with rasterio.open(prediction_path, "w", **image_meta) as dest:
                            dest.write(prediction, 1)

                        with rasterio.open(error_path, "w", **image_meta) as dest:
                            error = np.abs(prediction - target)
                            dest.write(error, 1)

                        probabilities_meta = image_meta.copy()
                        probabilities_meta.update({"dtype": "float32", "nodata": -1.0})
                        with rasterio.open(
                            probabilities_path, "w", **probabilities_meta
                        ) as dest:
                            dest.write(probability, 1)

                # Merge the prediction and error files
                merge_tiff_files(
                    prediction_files, current_prediction_folder.joinpath("merged.tif")
                )
                merge_tiff_files(
                    error_files, current_error_folder.joinpath("merged.tif")
                )
                merge_tiff_files(
                    probabilities_files,
                    current_probabilities_folder.joinpath("merged.tif"),
                )
