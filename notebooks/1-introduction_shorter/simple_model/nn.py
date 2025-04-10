import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from affine import Affine
from matplotlib import pyplot as plt

from simple_model.dataloader import ImagesLoader, TrainValLoaders
from simple_model.dataparse import merge_tiff_files
from simple_model.metrics import EvaluationMetrics, full_loss
from simple_model.model_plotting import (
    LayerDrawer,
    TensorDrawer,
    output_shape_conv,
    output_shape_pool,
    output_shape_upsample,
    plot_model,
)
from simple_model.visualisation import TrainingMetrics

# from tqdm.autonotebook import tqdm


# Function to check if running inside a Jupyter Notebook
def is_running_in_notebook():
    """Detect whether the script is running inside a Jupyter Notebook."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


# Import correct tqdm version
if is_running_in_notebook():
    from tqdm.notebook import tqdm  # Jupyter-friendly
else:
    from tqdm import tqdm  # Standard terminal tqdm


def get_new_model_name() -> str:
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, layers: int, reduce: bool):
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

        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(layers)])

        if reduce:
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        x_reduced = self.pool(x)
        # x_reduced = F.max_pool2d(x, kernel_size=2)

        return x, x_reduced


class Upsample(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, layers: int, upsample: bool
    ):
        super().__init__()

        if layers < 1:
            raise ValueError("Upsample layers must be at least 1")

        kernel_size = 3
        padding = kernel_size // 2

        self.convs = []

        for i in range(layers):
            if i == 0:
                if upsample:
                    in_channels_layer = 2 * in_channels
                else:
                    in_channels_layer = in_channels
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

        self.do_upsample = upsample

        if upsample:
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            x = self.upsample(x)
            x = torch.cat((x, y), dim=1)
        for conv in self.convs:
            x = F.relu(conv(x))

        return x


class SegmentationConvolutionalNetwork(nn.Module):

    def __init__(
        self,
        image_shape: tuple[int, int],
        encoder_channels: list[int],
        layers_downsample: int,
        layers_upsample: int,
        model_folder: Path,
        data_folders: list[Path],
        input_channels: int = 1,
    ):
        super().__init__()

        number_downsizing = len(encoder_channels) - 1
        divisor = 2**number_downsizing
        if image_shape[0] % divisor != 0 or image_shape[1] % divisor != 0:
            raise ValueError(
                f"With {number_downsizing + 1} encoder steps (= len(encoder_channels)) \
                                , the model performs {number_downsizing} downsampling, meaning that \
                                the width and height of the image should be divisible by \
                                2^{number_downsizing} (= {2**number_downsizing}) "
            )

        self.image_shape = image_shape
        self.input_channels = input_channels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder
        encoder_steps = [input_channels, *encoder_channels]
        encoders = []
        for i in range(len(encoder_steps) - 1):
            if i == len(encoder_steps) - 2:
                reduce = False
            else:
                reduce = True
            encoders.append(
                Downsample(
                    encoder_steps[i],
                    encoder_steps[i + 1],
                    layers_downsample,
                    reduce=reduce,
                )
            )
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoder_steps = encoder_steps[:]
        decoder_steps[0] = 1
        decoders = []
        for i in range(len(decoder_steps) - 2, -1, -1):
            if i == len(decoder_steps) - 2:
                upsample = False
            else:
                upsample = True
            decoders.append(
                Upsample(
                    decoder_steps[i + 1],
                    decoder_steps[i],
                    layers_upsample,
                    upsample=upsample,
                )
            )

        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv2d(decoder_steps[0], 1, kernel_size=1)

        self.to(self.device)

        # Save the model input parameters
        self.model_folder = model_folder
        parameters = {
            "image_shape": image_shape,
            "encoder_channels": encoder_channels,
            "layers_downsample": layers_downsample,
            "layers_upsample": layers_upsample,
            "input_channels": input_channels,
            "data_folders": list(map(str, data_folders)),
        }
        model_folder.mkdir(parents=True, exist_ok=True)
        parameters_path = model_folder / "parameters.json"
        with open(parameters_path, "w") as f:
            json.dump(parameters, f)

        self.training_folder = model_folder / "training"
        self.training_folder.mkdir(parents=True, exist_ok=True)

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
        metrics: EvaluationMetrics | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        logits = self.forward(images)
        loss, loss_components = full_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics.add_logits_and_targets(logits, targets)

        return loss, loss_components

    def validation_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        metrics: EvaluationMetrics | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        logits = self.forward(images)
        loss, loss_components = full_loss(logits, targets)

        with torch.no_grad():
            metrics.add_logits_and_targets(logits, targets)

        return loss, loss_components

    def _masks_from_original_sizes(
        self, targets: torch.Tensor, original_sizes: torch.Tensor
    ):
        masks = torch.zeros(targets.shape, dtype=torch.bool)
        for i, (width, height) in enumerate(original_sizes):
            masks[i, :, :height, :width] = 1

    def train_model(
        self,
        dataloaders: TrainValLoaders,
        max_epochs: int,
        initial_learning_rate: float,
        optimizer_type: str,
        stop_early_after: int = 10,
        save_weights: bool = True,
    ):
        print("Training the model...")
        print(f"Using the following device: {self.device}")

        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=initial_learning_rate, weight_decay=1e-4
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=initial_learning_rate,
                weight_decay=1e-5,
                momentum=0.9,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=stop_early_after // 2
        )
        progress = tqdm(range(max_epochs))

        # Save the training parameters
        training_parameters = {
            "max_epochs": max_epochs,
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
        training_parameters_path = self.model_folder / "training_parameters.json"
        with open(training_parameters_path, "w") as f:
            json.dump(training_parameters, f)

        # Best model and early stopping
        big_change_factor = 0.995
        best_loss_big_change = float("inf")
        epoch_big_change = 0
        best_loss = float("inf")
        best_model = None

        # Visualisation
        training_metrics = TrainingMetrics(show=True)
        training_metrics_vis_path = self.training_folder / "training_metrics_vis.png"
        training_metrics.visualise(save_paths=[training_metrics_vis_path])
        training_metrics_values_path = (
            self.training_folder / "training_metrics_values.json"
        )

        for epoch in progress:
            losses = dict(training=[], validation=[])
            bces = dict(training=[], validation=[])
            dices = dict(training=[], validation=[])
            focals = dict(training=[], validation=[])
            batches_sizes = dict(training=[], validation=[])

            train_eval_metrics = EvaluationMetrics()
            eval_eval_metrics = EvaluationMetrics()

            self.train()

            for (
                images,
                targets,
                indices,
                # original_sizes,
                # images_set_indices,
            ) in dataloaders.train_dataloader:
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                loss, (bce, dice, focal) = self.training_step(
                    images, targets, optimizer, metrics=train_eval_metrics
                )
                losses["training"].append(loss.item())
                bces["training"].append(bce.item())
                dices["training"].append(dice.item())
                focals["training"].append(focal.item())

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
                for (
                    images,
                    targets,
                    indices,
                    # original_sizes,
                    # images_set_indices,
                ) in dataloaders.val_dataloader:
                    images = images.to(self.device)
                    targets = targets.float().to(self.device)
                    loss, (bce, dice, focal) = self.validation_step(
                        images, targets, metrics=eval_eval_metrics
                    )
                    losses["validation"].append(loss.item())
                    bces["validation"].append(bce.item())
                    dices["validation"].append(dice.item())
                    focals["validation"].append(focal.item())

                    batches_sizes["validation"].append(images.shape[0])

            mean_validation_loss = (
                np.sum(
                    np.array(losses["validation"])
                    * np.array(batches_sizes["validation"])
                )
                / sum(batches_sizes["validation"])
            ).item()

            progress.set_description(
                "[training.loss] {:.6f}, [validation.loss] {:.6f}".format(
                    mean_training_loss, mean_validation_loss
                ),
                refresh=True,
            )

            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            # Update the visualisation with metrics values
            categories_names = ["Training", "Validation"]
            losses = {
                "Loss": [mean_training_loss, mean_validation_loss],
                "BCE Loss": [np.mean(bces["training"]), np.mean(bces["validation"])],
                "Dice Loss": [np.mean(dices["training"]), np.mean(dices["validation"])],
                "Focal Loss": [
                    np.mean(focals["training"]),
                    np.mean(focals["validation"]),
                ],
            }
            categories_eval_metrics = [train_eval_metrics, eval_eval_metrics]
            for i, (cat_name, eval_metric) in enumerate(
                zip(categories_names, categories_eval_metrics)
            ):
                for metric_name, metric_value in losses.items():
                    training_metrics.update(
                        category_name=cat_name,
                        metric_name=metric_name,
                        val=metric_value[i],
                    )
                for (
                    metric_name,
                    metric_value,
                ) in eval_metric.get_global_metrics().items():
                    training_metrics.update(
                        category_name=cat_name,
                        metric_name=metric_name,
                        val=metric_value,
                        y_limits=(0, 1),
                    )

            training_metrics.update(
                category_name="Training",
                metric_name="Learning rate",
                val=current_lr,
                y_axis="Learning rate",
            )

            training_metrics.end_loop(epoch=epoch)
            training_metrics.visualise(save_paths=[training_metrics_vis_path])
            training_metrics.save_metrics(save_path=training_metrics_values_path)

            if mean_validation_loss < best_loss:
                best_loss = mean_validation_loss
                best_model = self.state_dict()
                if save_weights:
                    self.save_weights()

            if mean_validation_loss < best_loss_big_change * big_change_factor:
                best_loss_big_change = mean_validation_loss
                epoch_big_change = epoch
            else:
                if epoch - epoch_big_change > stop_early_after:
                    break

        self.load_state_dict(best_model)

    def save_predictions(
        self,
        images_loader: ImagesLoader,
        dataloaders: TrainValLoaders,
        output_folder: Path,
        threshold: float = 0.5,
    ):
        # output_folder.mkdir(parents=True, exist_ok=True)

        dataloaders_postfixes = {
            "train": dataloaders.train_dataloader,
            "val": dataloaders.val_dataloader,
        }

        def get_path(
            category: str, postfix: str, image_set_name: str, tile_name: str = None
        ):
            if tile_name is not None:
                path = (
                    output_folder
                    / image_set_name
                    / f"tiles_{postfix}"
                    / category
                    / tile_name
                )
            else:
                path = output_folder / image_set_name / f"{category}_{postfix}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        self.eval()
        with torch.no_grad():
            for postfix, dataloader in dataloaders_postfixes.items():
                # Store files paths for merging
                predictions_files = defaultdict(list)
                errors_files = defaultdict(list)
                probabilities_files = defaultdict(list)

                for (
                    images,
                    targets,
                    indices,
                ) in dataloader:
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

                    for (
                        probability,
                        prediction,
                        target,
                        index,
                    ) in zip(
                        probabilities,
                        predictions,
                        targets,
                        indices,
                    ):
                        original_size = images_loader.get_original_size(index)
                        image_set_name = images_loader.get_image_set_name(index)

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
                        prediction_path = get_path(
                            "predictions", postfix, image_set_name, image_file_name
                        )
                        error_path = get_path(
                            "errors", postfix, image_set_name, image_file_name
                        )
                        probabilities_path = get_path(
                            "probabilities", postfix, image_set_name, image_file_name
                        )

                        predictions_files[image_set_name].append(prediction_path)
                        errors_files[image_set_name].append(error_path)
                        probabilities_files[image_set_name].append(probabilities_path)

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
                for image_set_name in predictions_files.keys():
                    full_predictions_path = get_path(
                        "predictions", postfix, image_set_name
                    )
                    full_errors_path = get_path("errors", postfix, image_set_name)
                    full_probabilities_path = get_path(
                        "probabilities", postfix, image_set_name
                    )
                    merge_tiff_files(
                        predictions_files[image_set_name], full_predictions_path
                    )
                    merge_tiff_files(errors_files[image_set_name], full_errors_path)
                    merge_tiff_files(
                        probabilities_files[image_set_name], full_probabilities_path
                    )

    def save_metrics(
        self,
        images_loader: ImagesLoader,
        dataloaders: TrainValLoaders,
        metrics_folder: Path,
        threshold: float = 0.5,
    ):
        # metrics_folder.mkdir(parents=True, exist_ok=True)

        dataloaders_postfixes = {
            "train": dataloaders.train_dataloader,
            "val": dataloaders.val_dataloader,
        }

        def get_path(
            category: str, postfix: str, image_set_name: str, tile_name: str = None
        ):
            if tile_name is not None:
                path = (
                    metrics_folder
                    / image_set_name
                    / f"tiles_{postfix}"
                    / category
                    / tile_name
                )
            else:
                path = metrics_folder / image_set_name / f"{category}_{postfix}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        self.eval()
        for postfix, dataloader in dataloaders_postfixes.items():
            index_all = []
            evaluation_metrics = EvaluationMetrics()

            with torch.no_grad():
                for (
                    images,
                    targets,
                    indices,
                ) in dataloader:
                    images = images.to(self.device)
                    logits = self.forward(images)

                    logits = logits.squeeze().detach().cpu().numpy()
                    targets = targets.squeeze().detach().cpu().numpy()

                    for logit, target, index in zip(logits, targets, indices):
                        # Get the index and the confusion matrix values
                        index = index.item()
                        index_all.append(index)

                        evaluation_metrics.add_logits_and_targets(
                            logit, target, threshold=threshold
                        )

            metrics_individual_values = evaluation_metrics.get_individual_metrics()

            # metrics_tiles_folders = {
            #     (metric_name, image_set_name): metrics_folder
            #     / metric_name
            #     / image_set_name
            #     / f"tiles_{postfix}"
            #     for metric_name in metrics_individual_values.keys()
            #     for image_set_name in images_loader.get_all_image_set_names()
            # }
            # for metrics_tiles_folder in metrics_tiles_folders.values():
            #     metrics_tiles_folder.mkdir(parents=True, exist_ok=True)

            for metric_name, metric_values in zip(
                metrics_individual_values.keys(), metrics_individual_values.values()
            ):
                metric_set_files = {
                    image_set_name: []
                    for image_set_name in images_loader.get_all_image_set_names()
                }
                for index, metric_value in zip(index_all, metric_values):
                    # Get the image set name
                    image_set_name = images_loader.get_image_set_name(index)

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
                    # metric_tiles_folder = metrics_tiles_folders[
                    #     (metric_name, image_set_name)
                    # ]
                    # metric_path = metric_tiles_folder / image_file_name
                    metric_path = get_path(
                        metric_name.lower(), postfix, image_set_name, image_file_name
                    )

                    with rasterio.open(metric_path, "w", **image_meta) as dest:
                        dest.write(np.array([metric_value]).reshape((1, 1)), 1)

                    metric_set_files[image_set_name].append(metric_path)

                # Merge the metrics files
                for image_set_name, metric_files in metric_set_files.items():
                    full_metric_path = get_path(
                        metric_name.lower(), postfix, image_set_name
                    )

                    merge_tiff_files(metric_files, full_metric_path)

            # Calculate the global metrics
            metrics_global_values = evaluation_metrics.get_global_metrics()

            metrics_path = metrics_folder / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_global_values, f)

    def save_weights(self):
        weights_path = self.model_folder / "weights.pth"
        torch.save(self.state_dict(), weights_path)

    def load_weights(self, model_folder: Path):
        weights_path = model_folder / "weights.pth"
        self.load_state_dict(torch.load(weights_path))

    @staticmethod
    def load_model(model_folder: Path, new_model_folder: Path | None = None):
        if new_model_folder is None:
            new_model_folder = model_folder

        new_model_folder.mkdir(parents=True, exist_ok=True)
        parameters_path = model_folder / "parameters.json"
        with open(parameters_path, "r") as f:
            parameters = json.load(f)

        model = SegmentationConvolutionalNetwork(
            **parameters, model_folder=new_model_folder
        )
        model.load_weights(model_folder)

        return model

    def plot_model(self, only_main_layers: bool = False):
        """Plot the model architecture."""
        tensors_drawers = []
        tensors_drawers.append(TensorDrawer(self.input_channels, self.image_shape[0]))

        layers_drawers = []
        input_shape = self.image_shape

        layers_categories = {
            "Conv2d": "Convolution",
            "BatchNorm2d": "Normalization",
            "MaxPool2d": "Max Pooling",
            "Upsample": "Upsampling",
        }
        layers_colors = {
            "Convolution": "blue",
            "Normalization": "orange",
            "Max Pooling": "green",
            "Upsampling": "purple",
            "Concatenation": "red",
        }

        encoder_output_per_shape = {}
        encoder_channel_per_shape = {}
        previous_channels = self.input_channels

        for encoder in self.encoders:
            for conv, bn in zip(encoder.convs, encoder.bns):
                # Calculate output shape of the convolutional layer
                output_shape = output_shape_conv(conv, input_shape)

                # Conv
                tensors_drawers.append(TensorDrawer(conv.out_channels, output_shape[0]))
                layers_drawers.append(
                    LayerDrawer(
                        layers_categories[conv.__class__.__name__],
                        tensors_drawers[-2],
                        tensors_drawers[-1],
                    )
                )

                # BatchNorm
                if not only_main_layers:
                    tensors_drawers.append(
                        TensorDrawer(bn.num_features, output_shape[0])
                    )
                    layers_drawers.append(
                        LayerDrawer(
                            layers_categories[bn.__class__.__name__],
                            tensors_drawers[-2],
                            tensors_drawers[-1],
                        )
                    )

                input_shape = output_shape

            encoder_output_per_shape[output_shape] = tensors_drawers[-1]
            encoder_channel_per_shape[output_shape] = conv.out_channels
            previous_channels = conv.out_channels

            # MaxPool2D
            output_shape = output_shape_pool(encoder.pool, input_shape)
            if isinstance(encoder.pool, nn.Identity):
                break
            tensors_drawers.append(TensorDrawer(conv.out_channels, output_shape[0]))
            layers_drawers.append(
                LayerDrawer(
                    layers_categories[encoder.pool.__class__.__name__],
                    tensors_drawers[-2],
                    tensors_drawers[-1],
                )
            )

            input_shape = output_shape

        for decoder in self.decoders:
            # Upsample
            output_shape = output_shape_upsample(decoder.upsample, input_shape)
            if not isinstance(decoder.upsample, nn.Identity):
                tensors_drawers.append(TensorDrawer(conv.out_channels, output_shape[0]))
                layers_drawers.append(
                    LayerDrawer(
                        layers_categories[decoder.upsample.__class__.__name__],
                        tensors_drawers[-2],
                        tensors_drawers[-1],
                    )
                )
                input_shape = output_shape

            # Concatenate
            if not isinstance(decoder.upsample, nn.Identity):
                tensors_drawers.append(
                    TensorDrawer(
                        previous_channels + encoder_channel_per_shape[output_shape],
                        output_shape[0],
                    )
                )
                layers_drawers.append(
                    LayerDrawer(
                        "Concatenation",
                        [
                            encoder_output_per_shape[output_shape],
                            tensors_drawers[-2],
                        ],
                        tensors_drawers[-1],
                    )
                )

            for conv in decoder.convs:
                # Calculate output shape of the convolutional layer
                output_shape = output_shape_conv(conv, input_shape)

                # Conv
                tensors_drawers.append(TensorDrawer(conv.out_channels, output_shape[0]))
                layers_drawers.append(
                    LayerDrawer(
                        layers_categories[conv.__class__.__name__],
                        tensors_drawers[-2],
                        tensors_drawers[-1],
                    )
                )

                input_shape = output_shape
                previous_channels = conv.out_channels

        # Set a color for each size of tensor
        tensors_colors = {}
        for i, tensor_drawer in enumerate(tensors_drawers):
            if tensor_drawer.image_size not in tensors_colors:
                # Get a light color from matplotlib colors
                tensors_colors[tensor_drawer.image_size] = plt.cm.tab10(
                    len(tensors_colors) % 10
                )[:3]

        plot_model(
            tensors_drawers,
            layers_drawers,
            tensors_colors,
            layers_colors,
            self.model_folder / "model_plot.png",
        )
