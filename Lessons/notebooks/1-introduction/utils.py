from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim.adam import Adam
from torchvision import transforms
from tqdm.notebook import tqdm


def plot_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray,
    class_names: list[str],
    ax: Axes | None = None,
):
    conf_matrix = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual Class")
    ax.set_xlabel("Predicted Class")


def plot_evaluation(
    y_true: NDArray,
    y_pred: NDArray,
    class_names: list[str],
    ax: Axes | None = None,
):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.unique(y_true)
    )

    x = np.arange(len(class_names))
    width = 0.25

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    rects1 = ax.bar(x - width, precision, width, label="Precision")
    rects2 = ax.bar(x, recall, width, label="Recall")
    rects3 = ax.bar(x + width, f1, width, label="F1-Score")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Scores")
    ax.set_title("Precision, Recall, and F1-Score by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()

    # Attach a label above each bar in the barplot
    def autolabel(rects):
        """Attach a text label above each bar, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)


def plot_decision_boundaries(
    model,
    X_test: NDArray,
    y_test: NDArray,
    class_names: list[str],
    ax: Axes | None = None,
):
    # Create a meshgrid for visualization
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict over the meshgrid to get decision boundaries
    Z = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")

    # Plot the training points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, s=50, edgecolor="k", cmap="viridis"
    )
    ax.set_title("Decision Boundaries")
    ax.set_xlabel(class_names[0])
    ax.set_ylabel(class_names[1])


def plot_full_evaluation(
    y_true: NDArray, y_pred: NDArray, class_names: list[str], title: str | None = None
):
    accuracy = accuracy_score(y_true, y_pred)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    if title is not None:
        title = f"{title} - "
    plt.suptitle(f"{title}Accuracy: {accuracy:.4f}")
    plot_confusion_matrix(y_true, y_pred, class_names, ax=axs[0])
    plot_evaluation(y_true, y_pred, class_names, ax=axs[1])


class ChannelActivation(nn.Module):
    def __init__(self, model: nn.Module, layer: int, channel: int, device):
        super(ChannelActivation, self).__init__()
        self.layer = layer
        self.channel = channel
        self.model = model.to(device)
        self.model.eval()

    def forward(self, x):
        for i in range(self.layer + 1):
            x = self.model.layers[i](x)
        output_channel = x[:, self.channel]
        mean_activation = torch.mean(output_channel)
        return mean_activation


def create_activation_image(
    model: nn.Module,
    layer: int,
    channel: int,
    input_mean: float,
    input_std: float,
    lr: float = 0.02,
    steps: int = 200,
    show_steps: bool = False,
    transform: transforms.Compose | None = None,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    progress_bar: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    input_image_cpu = (
        input_std * torch.randn((1, *input_shape)) + input_mean
    ).requires_grad_()

    if show_steps:
        saved_steps = np.unique(
            (np.array([0, 0.05, 0.1, 0.17, 0.3, 0.45, 0.62, 0.8, 1]) * steps).astype(
                int
            )
        )
        saved_images = [np.copy(input_image_cpu.detach())]

    model = ChannelActivation(model, layer, channel, device).to(device)
    optimizer = Adam([input_image_cpu], lr=lr, maximize=True)

    model.eval()
    iterator = range(1, steps + 1)
    if progress_bar:
        iterator = tqdm(iterator, desc="step")

    for step in iterator:
        input_image_gpu = input_image_cpu.to(device)
        if transform is not None:
            input_image_gpu = transform(input_image_gpu)

        optimizer.zero_grad()
        mean_activation = model(input_image_gpu).to(device)
        mean_activation.backward()
        optimizer.step()

        # Clip the value of the pixels to avoid saturation
        input_image_cpu.data = torch.clamp(input_image_cpu.data, 0.1, 0.9)

        # Store the new image
        if show_steps and (step == saved_steps[len(saved_images)]):
            saved_images.append(np.copy(input_image_cpu.detach()))

    # Show the image at different steps
    if show_steps:
        instances = len(saved_steps)
        cols = int(np.ceil(np.sqrt(instances)))
        rows = int(np.ceil(instances / cols))
        plt.figure(figsize=(15, 15 * rows / cols))
        for i, (step, image) in enumerate(zip(saved_steps, saved_images)):
            plt.subplot(rows, cols, i + 1)
            cmap = "gray" if input_shape[0] == 1 else None
            plt.imshow(image[0].transpose(1, 2, 0), cmap=cmap)
            plt.title(f"Step {step}")
            plt.axis("off")
        plt.show()

    return input_image_cpu[0].detach().numpy()
