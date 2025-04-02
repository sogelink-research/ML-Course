import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from affine import Affine
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm.autonotebook import tqdm

from simple_model.dataloader import ImagesLoader
from simple_model.dataparse import merge_tiff_files
from simple_model.visualisation import TrainingMetrics

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


def accuracy(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the accuracy metric (TP + TN) / (TP + FP + TN + FN).
    """
    return (tp + tn) / (tp + fp + tn + fn)


def precision(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the precision metric TP / (TP + FP).
    If the denominator is 0, return -1.0.
    """
    return np.where(tp + fp == 0, -1.0, tp / (tp + fp))


def recall(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """
    Calculate the recall metric TP / (TP + FN).
    If the denominator is 0, return -1.0.
    """
    return np.where(tp + fn == 0, -1.0, tp / (tp + fn))


def f1(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """
    Calculate the F1 score metric 2 * (precision * recall) / (precision + recall).
    If the denominator is 0, return -1.0.
    """
    precision_value = precision(tp, fp, tn, fn)
    recall_value = recall(tp, fp, tn, fn)
    return np.where(
        precision_value + recall_value == 0,
        -1.0,
        2 * (precision_value * recall_value) / (precision_value + recall_value),
    )


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


def full_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

    # Combine the losses
    total = alpha * bce + beta * dice + gamma * focal
    return total


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, layers: int):
        super().__init__()

        kernel_size = 3
        padding = kernel_size // 2

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
                for i in range(layers)
            ]
        )

        # self.conv1 = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=kernel_size, padding=padding
        # )
        # self.conv2 = nn.Conv2d(
        #     out_channels, out_channels, kernel_size=kernel_size, padding=padding
        # )

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x_reduced = F.max_pool2d(x, kernel_size=2)

        return x, x_reduced


class Upsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, layers: int):
        super().__init__()

        if layers < 1:
            raise ValueError("Upsample layers must be at least 1")

        kernel_size = 3
        padding = kernel_size // 2

        self.convs = []

        for i in range(layers):
            if i == 0:
                in_channels_layer = 2 * in_channels
            elif i == 1:
                in_channels_layer = in_channels
            else:
                in_channels_layer = out_channels

            if i == 0 and layers > 1:
                out_channels_layer = in_channels
            else:
                out_channels_layer = out_channels

            self.convs.append(
                nn.Conv2d(
                    in_channels_layer,
                    out_channels_layer,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        self.convs = nn.ModuleList(self.convs)

        # self.conv1 = nn.Conv2d(
        #     2 * in_channels, in_channels, kernel_size=kernel_size, padding=padding
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels, out_channels, kernel_size=kernel_size, padding=padding
        # )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat((x, y), dim=1)
        for conv in self.convs:
            x = F.relu(conv(x))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))

        return x


class SegmentationConvolutionalNetwork(nn.Module):

    def __init__(
        self,
        image_size: tuple[int, int],
        encoder_channels: list[int],
        layers_downsample: int,
        layers_upsample: int,
        model_folder: Path,
        data_folders: list[Path],
        input_channels: int = 1,
    ):
        super().__init__()

        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        encoder_steps = [input_channels, *encoder_channels]
        self.encoders = nn.ModuleList(
            [
                Downsample(encoder_steps[i], encoder_steps[i + 1], layers_downsample)
                for i in range(len(encoder_steps) - 1)
            ]
        )

        # Decoder
        decoder_steps = encoder_steps[:]
        decoder_steps[0] = 1
        self.decoders = nn.ModuleList(
            [
                Upsample(decoder_steps[i + 1], decoder_steps[i], layers_upsample)
                for i in range(len(encoder_steps) - 2, -1, -1)
            ]
        )

        self.final_conv = nn.Conv2d(decoder_steps[0], 1, kernel_size=1)

        self.to(self.device)

        # Save the model input parameters
        self.model_folder = model_folder
        parameters = {
            "image_size": image_size,
            "encoder_channels": encoder_channels,
            "layers_downsample": layers_downsample,
            "layers_upsample": layers_upsample,
            "input_channels": input_channels,
            "data_folders": list(map(str, data_folders)),
        }
        model_folder.mkdir(parents=True, exist_ok=True)
        parameters_path = model_folder.joinpath("parameters.json")
        with open(parameters_path, "w") as f:
            json.dump(parameters, f)

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
    ) -> float:
        logits = self.forward(images)
        loss = full_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.nanmean().item()

    def validation_step(self, images: torch.Tensor, targets: torch.Tensor) -> float:
        logits = self.forward(images)
        loss = full_loss(logits, targets)

        return loss.nanmean().item()

    def _masks_from_original_sizes(
        self, targets: torch.Tensor, original_sizes: torch.Tensor
    ):
        masks = torch.zeros(targets.shape, dtype=torch.bool)
        for i, (width, height) in enumerate(original_sizes):
            masks[i, :, :height, :width] = 1

    def train_model(
        self,
        dataloaders: tuple[torch.utils.data.DataLoader],
        epochs: int,
        visualisation_output: Path,
        stop_early_after: int = 10,
    ):
        print("Training the model...")
        print(f"Using the following device: {self.device}")

        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        progress = tqdm(range(epochs))

        # Save the training parameters
        training_parameters = {
            "epochs": epochs,
            "stop_early_after": stop_early_after,
            "optimizer": {
                "name": optimizer.__class__.__name__,
                "params": optimizer.state_dict(),
            },
            "scheduler": {
                "name": scheduler.__class__.__name__,
                "params": scheduler.state_dict(),
            },
        }
        training_parameters_path = self.model_folder.joinpath(
            "training_parameters.json"
        )
        with open(training_parameters_path, "w") as f:
            json.dump(training_parameters, f)

        # Best model and early stopping
        big_change_factor = 0.995
        best_loss_big_change = float("inf")
        epoch_big_change = 0
        best_loss = float("inf")
        best_model = None

        # Visualisation
        # visualisation = TrainingVisualisation()
        visualisation = TrainingMetrics(show=True)
        visualisation.visualise()

        all_mean_losses = dict(training=[], validation=[])

        for epoch in progress:
            losses = dict(training=[], validation=[])
            batches_sizes = dict(training=[], validation=[])

            self.train()

            for images, targets, indices, original_sizes in dataloaders[0]:
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                losses["training"].append(
                    self.training_step(images, targets, optimizer)
                )
                batches_sizes["training"].append(images.shape[0])

            mean_training_loss = (
                np.sum(
                    np.array(losses["training"]) * np.array(batches_sizes["training"])
                )
                / sum(batches_sizes["training"])
            ).item()
            if scheduler is not None:
                scheduler.step(mean_training_loss)

            with torch.no_grad():
                self.eval()

                for images, targets, indices, original_sizes in dataloaders[1]:
                    images = images.to(self.device)
                    targets = targets.float().to(self.device)
                    losses["validation"].append(self.validation_step(images, targets))
                    batches_sizes["validation"].append(images.shape[0])

            mean_validation_loss = (
                np.sum(
                    np.array(losses["validation"])
                    * np.array(batches_sizes["validation"])
                )
                / sum(batches_sizes["validation"])
            ).item()

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
            # visualisation.update_plots(
            #     epoch=epoch,
            #     train_loss=mean_training_loss,
            #     val_loss=mean_validation_loss,
            #     lr=current_lr,
            # )
            visualisation.update(
                category_name="Training",
                metric_name="Loss",
                val=mean_training_loss,
                y_axis="Loss",
            )
            visualisation.update(
                category_name="Validation",
                metric_name="Loss",
                val=mean_validation_loss,
                y_axis="Loss",
            )
            visualisation.update(
                category_name="Training",
                metric_name="Learning rate",
                val=current_lr,
                y_axis="Learning rate",
            )

            visualisation.end_loop(epoch=epoch)
            visualisation.visualise()

            if mean_validation_loss < best_loss:
                best_loss = mean_validation_loss
                best_model = self.state_dict()

            if mean_validation_loss < best_loss_big_change * big_change_factor:
                best_loss_big_change = mean_validation_loss
                epoch_big_change = epoch
            else:
                if epoch - epoch_big_change > stop_early_after:
                    break

        visualisation_output.parent.mkdir(parents=True, exist_ok=True)
        # visualisation.savefig(visualisation_output)
        # visualisation.close()
        visualisation.save_metrics(visualisation_output)

        self.load_state_dict(best_model)

    def save_predictions(
        self,
        images_loader: ImagesLoader,
        dataloaders: tuple[torch.utils.data.DataLoader],
        output_folder: Path,
        threshold: float = 0.5,
    ):
        output_folder.mkdir(parents=True, exist_ok=True)
        predictions_folder = output_folder / "output"
        errors_folder = output_folder / "error"
        probabilities_folder = output_folder / "probabilities"

        train_postfix = "train"
        val_postfix = "val"

        self.eval()
        with torch.no_grad():
            for dataloader, postfix in zip(dataloaders, [train_postfix, val_postfix]):
                # Create subfolders for predictions and errors
                current_predictions_tiles_folder = (
                    predictions_folder / postfix / "tiles"
                )
                current_predictions_tiles_folder.mkdir(parents=True, exist_ok=True)

                current_errors_tiles_folder = errors_folder / postfix / "tiles"
                current_errors_tiles_folder.mkdir(parents=True, exist_ok=True)

                current_probabilities_tiles_folder = (
                    probabilities_folder / postfix / "tiles"
                )
                current_probabilities_tiles_folder.mkdir(parents=True, exist_ok=True)

                # Store files paths for merging
                predictions_files = []
                errors_files = []
                probabilities_files = []

                for images, targets, indices, original_sizes in dataloader:
                    images = images.to(self.device)
                    logits = self.forward(images)
                    probabilities = torch.sigmoid(logits)

                    probabilities = probabilities.squeeze().detach().cpu().numpy()
                    targets = targets.squeeze().detach().cpu().numpy()

                    # Transpose to write the image with rasterio
                    probabilities = np.transpose(probabilities, (0, 2, 1))
                    targets = np.transpose(targets, (0, 2, 1))

                    # Set predictions to 0 or 1
                    predictions = np.where(probabilities > threshold, 1, 0)

                    for probability, prediction, target, index, original_size in zip(
                        probabilities, predictions, targets, indices, original_sizes
                    ):
                        # Resize the predictions to the original size
                        probability = probability[
                            : original_size[1], : original_size[0]
                        ]
                        prediction = prediction[: original_size[1], : original_size[0]]
                        target = target[: original_size[1], : original_size[0]]

                        index = index.item()
                        image_file_name = images_loader.get_file_name(index)
                        image_meta = images_loader.get_meta(index).copy()

                        # Get the file paths
                        prediction_path = current_predictions_tiles_folder.joinpath(
                            image_file_name
                        )
                        error_path = current_errors_tiles_folder.joinpath(
                            image_file_name
                        )
                        probabilities_path = (
                            current_probabilities_tiles_folder.joinpath(image_file_name)
                        )
                        predictions_files.append(prediction_path)
                        errors_files.append(error_path)
                        probabilities_files.append(probabilities_path)

                        image_meta.update({"dtype": "uint8", "nodata": 255})
                        with rasterio.open(prediction_path, "w", **image_meta) as dest:
                            dest.write(prediction, 1)

                        with rasterio.open(error_path, "w", **image_meta) as dest:
                            error = prediction * 2 + target
                            dest.write(error, 1)

                        probabilities_meta = image_meta.copy()
                        probabilities_meta.update({"dtype": "float32", "nodata": -1.0})
                        with rasterio.open(
                            probabilities_path, "w", **probabilities_meta
                        ) as dest:
                            dest.write(probability, 1)

                # Merge the prediction and error files
                merge_tiff_files(
                    predictions_files, output_folder / f"predictions_{postfix}.tif"
                )
                merge_tiff_files(errors_files, output_folder / f"errors_{postfix}.tif")
                merge_tiff_files(
                    probabilities_files, output_folder / f"probabilities_{postfix}.tif"
                )

    def save_metrics(
        self,
        images_loader: ImagesLoader,
        dataloaders: tuple[torch.utils.data.DataLoader],
        metrics_folder: Path,
        threshold: float = 0.5,
    ):
        metrics_folder.mkdir(parents=True, exist_ok=True)

        metrics_names = ["accuracy", "precision", "recall", "f1"]
        metrics_functions = [accuracy, precision, recall, f1]

        train_postfix = "train"
        val_postfix = "val"

        self.eval()
        for dataloader, postfix in zip(dataloaders, [train_postfix, val_postfix]):
            index_all = []
            tp_all = []
            fp_all = []
            tn_all = []
            fn_all = []

            metrics_tiles_folders = [
                metrics_folder / name / f"tiles_{postfix}" for name in metrics_names
            ]

            with torch.no_grad():
                for images, targets, indices, original_sizes in dataloader:
                    images = images.to(self.device)
                    logits = self.forward(images)
                    probabilities = torch.sigmoid(logits)

                    probabilities = probabilities.squeeze().detach().cpu().numpy()
                    targets = targets.squeeze().detach().cpu().numpy()

                    # Transpose to write the image with rasterio
                    probabilities = np.transpose(probabilities, (0, 2, 1))
                    targets = np.transpose(targets, (0, 2, 1))

                    # Set predictions to 0 or 1
                    predictions = np.where(probabilities > threshold, 1, 0)

                    for prediction, target, index in zip(predictions, targets, indices):
                        # Resize the predictions to the original size
                        prediction = prediction[
                            : original_sizes[0][1], : original_sizes[0][0]
                        ]
                        target = target[: original_sizes[0][1], : original_sizes[0][0]]

                        # Get the index and the confusion matrix values
                        index = index.item()
                        index_all.append(index)
                        tp = np.sum(np.logical_and(prediction == 1, target == 1))
                        fp = np.sum(np.logical_and(prediction == 1, target == 0))
                        tn = np.sum(np.logical_and(prediction == 0, target == 0))
                        fn = np.sum(np.logical_and(prediction == 0, target == 1))
                        tp_all.append(tp)
                        fp_all.append(fp)
                        tn_all.append(tn)
                        fn_all.append(fn)

            # Calculate and save metrics for each image
            for metric_name, metric_tiles_folder, metric_function in zip(
                metrics_names, metrics_tiles_folders, metrics_functions
            ):
                metric_tiles_folder.mkdir(parents=True, exist_ok=True)
                metrics = metric_function(
                    np.array(tp_all),
                    np.array(fp_all),
                    np.array(tn_all),
                    np.array(fn_all),
                )
                metric_files = []
                for index, metric in zip(index_all, metrics):
                    # Get the correct metadata to cover the image with one pixel
                    image_meta = images_loader.get_meta(index).copy()
                    image_meta["transform"] = Affine(
                        image_meta["transform"].a * image_meta["width"],
                        image_meta["transform"].b,
                        image_meta["transform"].c,
                        image_meta["transform"].d,
                        image_meta["transform"].e * image_meta["height"],
                        image_meta["transform"].f,
                    )
                    image_meta.update(
                        {"dtype": "float32", "nodata": -1.0, "width": 1, "height": 1}
                    )
                    if image_meta["width"] != image_meta["height"]:
                        print(image_meta)

                    image_file_name = images_loader.get_file_name(index)
                    metric_path = metric_tiles_folder.joinpath(image_file_name)

                    with rasterio.open(metric_path, "w", **image_meta) as dest:
                        dest.write(np.array([metric]).reshape((1, 1)), 1)

                    metric_files.append(metric_path)

                # Merge the metrics files
                merge_tiff_files(
                    metric_files,
                    metrics_folder.joinpath(f"{metric_name}_{postfix}.tif"),
                )

            # Calculate the global metrics
            tp_total = np.sum(tp_all)
            fp_total = np.sum(fp_all)
            tn_total = np.sum(tn_all)
            fn_total = np.sum(fn_all)
            metrics = [
                metric_function(tp_total, fp_total, tn_total, fn_total).item()
                for metric_function in metrics_functions
            ]
            metrics_dict = dict(zip(metrics_names, metrics))
            metrics_path = metrics_folder.joinpath("metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f)

    def save_weights(self):
        weights_path = self.model_folder.joinpath("weights.pth")
        torch.save(self.state_dict(), weights_path)

    def load_weights(self, model_folder: Path):
        weights_path = model_folder.joinpath("weights.pth")
        self.load_state_dict(torch.load(weights_path))

    @staticmethod
    def load_model(model_folder: Path, new_model_folder: Path | None = None):
        if new_model_folder is None:
            new_model_folder = model_folder

        new_model_folder.mkdir(parents=True, exist_ok=True)
        parameters_path = model_folder.joinpath("parameters.json")
        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        model = SegmentationConvolutionalNetwork(
            **parameters, model_folder=new_model_folder
        )
        model.load_weights(model_folder)

        return model
