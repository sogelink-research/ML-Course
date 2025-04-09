from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import nn


def draw_broken_arrow(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    label: str | None = None,
    color: str = "black",
    width: float = 1.0,
):
    """Draws a broken arrow with multiple segments."""
    head_length = width * 4
    real_last_x = xs[-1]
    real_last_y = ys[-1]
    # Draw the segments of the arrow
    if xs[-1] != xs[-2]:
        xs[-1] -= head_length * np.sign(xs[-1] - xs[-2]) / 2
    if ys[-1] != ys[-2]:
        ys[-1] -= head_length * np.sign(ys[-1] - ys[-2]) / 2
    for i in range(len(xs) - 1):
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            linestyle="-",
            color=color,
            lw=width,
        )

    # Draw the arrowhead
    ax.annotate(
        "",
        xy=(real_last_x, real_last_y),
        xytext=(xs[-2], ys[-2]),
        arrowprops=dict(
            color=color,
            # arrowstyle="->",
            # lw=width,
            # mutation_scale=5,
            # shrinkA=0,
            # shrinkB=0,
            headwidth=width * 3,
            headlength=head_length,
            width=0,
            shrink=0,
        ),
    )

    # Draw label with a box
    if label is not None:
        text = ax.text(
            xs[0],
            ys[0],
            label,
            fontsize=10,
            ha="center",
            va="center",
            color=color,
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"),
        )
    else:
        text = None
    return text


class TensorDrawer:

    def __init__(self, channels: int, image_size: int):
        self.channels = channels
        self.image_size = image_size

    def draw_rectangle(self, ax: plt.Axes, x_left: float, y_center: float, color: str):
        """Draw the rectangle representing the tensor."""

        self.x_left = x_left
        self.x_right = x_left + self.channels
        self.y_center = y_center
        self.y_bottom = y_center - self.image_size / 2
        self.y_top = y_center + self.image_size / 2

        rect = plt.Rectangle(
            (self.x_left, self.y_bottom),
            self.channels,
            self.image_size,
            color=color,
            alpha=0.6,
            edgecolor="black",
        )
        ax.add_patch(rect)

    def draw_shape(self, ax: plt.Axes, y: float):
        """Draw the shape representing the tensor."""
        ax.text(
            self.x_left + self.channels / 2,
            y,
            # f"{self.channels}×{self.image_size}²",
            f"{self.channels}",
            fontsize=10,
            ha="center",
            va="top",
            color="black",
        )


class LayerDrawer:

    def __init__(
        self,
        category: "str",
        input_tensors: TensorDrawer | list[TensorDrawer],
        output_tensor: TensorDrawer,
    ):
        self.category = category
        if isinstance(input_tensors, TensorDrawer):
            input_tensors = [input_tensors]
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor

    def draw_arrow(
        self,
        ax: plt.Axes,
        color: str,
    ):
        """Draw the layer as an arrow."""

        # Find the rightmost input tensor
        max_tensor_x_left = max(
            input_tensor.x_left for input_tensor in self.input_tensors
        )
        max_tensor_x_right = max(
            input_tensor.x_right for input_tensor in self.input_tensors
        )

        for input_tensor in self.input_tensors:
            # Create the path
            x_start = input_tensor.x_right
            x_end = self.output_tensor.x_left
            y_start = input_tensor.y_center
            y_end = self.output_tensor.y_center

            if input_tensor.x_right < max_tensor_x_right:
                # Get the y position to dodge
                y_over = self.output_tensor.y_top + 30
                x_turn = (x_start + max_tensor_x_left) / 2
                x_end = (self.output_tensor.x_left + self.output_tensor.x_right) / 2
                y_end = input_tensor.y_top

                # Draw the broken arrow
                xs = [x_start, x_turn, x_turn, x_end, x_end]
                ys = [y_start, y_start, y_over, y_over, y_end]

                draw_broken_arrow(ax, xs, ys, color=color, width=2)

            else:
                x_turn = (x_start + x_end) / 2 - 5
                xs = [x_start, x_turn, x_turn, x_end]
                ys = [y_start, y_start, y_end, y_end]

                draw_broken_arrow(ax, xs, ys, color=color, width=2)


def get_layers_legend(layers_drawers: list[LayerDrawer], layers_colors: dict):
    """Plot the legend for the layers."""

    # Keep only the used layer categories
    used_layers = set(layer_drawer.category for layer_drawer in layers_drawers)
    layers_colors = {
        layer: color for layer, color in layers_colors.items() if layer in used_layers
    }

    # Create legend elements
    legend_elements = []
    for layer, color in layers_colors.items():
        # legend_elements.append(
        #     plt.Line2D(
        #         [0],
        #         [0],
        #         marker="o",
        #         color="w",
        #         label=layer,
        #         markerfacecolor=color,
        #         markersize=10,
        #     )
        # )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=">",
                color="w",
                label=layer,
                markerfacecolor=color,
                markersize=10,
            ),
        )

    return legend_elements


def get_tensors_legend(tensors_drawers: list[TensorDrawer], tensors_colors: dict):
    """Plot the legend for the tensors."""

    # Keep only the used tensor sizes
    used_tensors = set(tensor_drawer.image_size for tensor_drawer in tensors_drawers)
    tensors_colors = {
        tensor: color
        for tensor, color in tensors_colors.items()
        if tensor in used_tensors
    }

    # Create legend elements
    legend_elements = []
    for tensor, color in tensors_colors.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"{tensor} channels",
                markerfacecolor=color,
                markersize=10,
            )
        )
    return legend_elements


def plot_legend(ax: plt.Axes, handles: list):
    """Plot the legend for the layers."""
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        title="Layers",
        title_fontsize=12,
    )


def plot_model(
    tensors_drawers: list[TensorDrawer],
    layers_drawers: list[LayerDrawer],
    tensors_colors: dict[int, str],
    layers_colors: dict[str, str],
    save_path: Path | None = None,
):
    """Plot the model architecture."""
    fig, ax = plt.subplots(figsize=(15, 10))

    x_offset = 20
    x_left = 0
    y_center = 0

    x_lefts = []
    x_rights = []
    y_centers = []
    min_y = float("inf")
    max_y = float("-inf")

    for i in range(len(tensors_drawers)):
        tensor_drawer = tensors_drawers[i]

        y_center = tensor_drawer.image_size

        color = tensors_colors.get(tensor_drawer.image_size, "gray")
        tensor_drawer.draw_rectangle(ax, x_left, y_center, color=color)

        x_lefts.append(tensor_drawer.x_left)
        x_rights.append(tensor_drawer.x_right)
        y_centers.append(tensor_drawer.y_center)
        min_y = min(min_y, tensor_drawer.y_bottom)
        max_y = max(max_y, tensor_drawer.y_top)

        x_left = x_rights[-1] + x_offset

        if i > 0:
            layer_drawer = layers_drawers[i - 1]
            layer_drawer.draw_arrow(
                ax,
                color=layers_colors[layer_drawer.category],
            )

    for i, tensor_drawer in enumerate(tensors_drawers):
        tensor_drawer.draw_shape(ax, min_y - 10)

    # Draw the legend
    handles = get_layers_legend(layers_drawers, layers_colors)
    handles += get_tensors_legend(tensors_drawers, tensors_colors)
    plot_legend(ax, handles)

    # Adapt figure size
    fig_width = x_rights[-1] - x_lefts[0]
    fig_height = max_y - min_y
    fig.set_size_inches(fig_width / 50, fig_height / 50)

    # Hide the axes
    ax.axis("off")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def output_shape_conv(conv: nn.Conv2d, input_height_width: tuple[int, int]):
    """Calculate the output shape of a Conv2d layer."""
    height, width = input_height_width

    kernel_size = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]

    out_height = (height - kernel_size + 2 * padding) // stride + 1
    out_width = (width - kernel_size + 2 * padding) // stride + 1

    return out_height, out_width


def output_shape_pool(
    pool: nn.MaxPool2d | nn.Identity, input_height_width: tuple[int, int]
):
    """Calculate the output shape of a MaxPool2d layer."""
    if isinstance(pool, nn.Identity):
        return input_height_width

    height, width = input_height_width

    kernel_size = pool.kernel_size
    stride = pool.stride
    padding = pool.padding

    out_height = (height - kernel_size + 2 * padding) // stride + 1
    out_width = (width - kernel_size + 2 * padding) // stride + 1

    return out_height, out_width


def output_shape_upsample(
    upsample: nn.Upsample | nn.Identity, input_height_width: tuple[int, int]
):
    """Calculate the output shape of an Upsample layer."""
    if isinstance(upsample, nn.Identity):
        return input_height_width

    height, width = input_height_width

    scale_factor = upsample.scale_factor

    out_height = int(height * scale_factor)
    out_width = int(width * scale_factor)

    return out_height, out_width
