import bisect
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from ipywidgets import Output
from matplotlib.ticker import MaxNLocator

# class TrainingVisualisation:

#     def __init__(self):
#         self.training_losses = []
#         self.validation_losses = []
#         self.learning_rates = []
#         self.keep_updating = True

#         # Create the figure once and update it dynamically
#         print("Turning interactive mode on...")
#         plt.ion()  # Turn on interactive mode
#         print("Interactive mode turned on...")
#         self.fig, self.ax_losses = plt.subplots(figsize=(10, 5))

#         # Twin axis for learning rate
#         self.ax_lr = self.ax_losses.twinx()

#         # Connect to close event
#         self.fig.canvas.mpl_connect("close_event", self.on_close)

#     def on_close(self, event):
#         """Callback to detect when the figure window is closed."""
#         print("Plot window closed. Stopping updates.")
#         self.keep_updating = False

#     def update_plots(self, epoch: int, train_loss: float, val_loss: float, lr: float):
#         """Update the plot with new data."""
#         if not self.keep_updating:
#             return

#         self.training_losses.append(train_loss)
#         self.validation_losses.append(val_loss)
#         self.learning_rates.append(lr)

#         # Clear the plot
#         self.ax_losses.clear()
#         self.ax_lr.clear()

#         # Plot Losses
#         current_epoch = len(self.training_losses)
#         epochs_range = range(1, current_epoch + 1)
#         if current_epoch > 1:
#             self.ax_losses.plot(
#                 epochs_range, self.training_losses, label="Train Loss", color="blue"
#             )
#             self.ax_losses.plot(
#                 epochs_range,
#                 self.validation_losses,
#                 label="Validation Loss",
#                 color="red",
#             )
#         else:
#             self.ax_losses.scatter(
#                 1, train_loss, label="Train Loss", color="blue", marker="o"
#             )
#             self.ax_losses.scatter(
#                 1, val_loss, label="Validation Loss", color="red", marker="o"
#             )
#         self.ax_losses.set_xlabel("Epoch")
#         self.ax_losses.set_ylabel("Loss")
#         self.ax_losses.legend(loc="upper left")
#         self.ax_losses.grid(True)

#         # Set X axis labels to integers
#         self.ax_losses.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

#         # Plot Learning Rate
#         if current_epoch > 1:
#             self.ax_lr.plot(
#                 epochs_range,
#                 self.learning_rates,
#                 label="Learning Rate",
#                 color="green",
#                 linestyle="dashed",
#             )
#         else:
#             self.ax_lr.scatter(1, lr, label="Learning Rate", color="green", marker="o")
#         self.ax_lr.set_ylabel("Learning Rate")
#         self.ax_lr.set_yscale("log")
#         self.ax_lr.legend(loc="upper right")

#         plt.title(f"Epoch {epoch}")
#         plt.tight_layout()

#         clear_output(wait=True)  # Clears previous output
#         display(self.fig)  # Displays updated figure in Jupyter

#         plt.draw()  # Update the plot
#         # plt.pause(0.1)  # Allow time for rendering

#     def savefig(self, filename: Path):
#         plt.savefig(filename)

#     def close(self):
#         plt.ioff()  # Turn off interactive mode when training is done
#         plt.show()  # Final show of the plot


class TrainingMetrics:
    def __init__(self, show: bool = True) -> None:
        self.show = show
        self.reset()
        self.fig_num = None

    def _create_fig_num(self) -> None:
        if self.fig_num is None:
            existing_figures = plt.get_fignums()
            if existing_figures:
                self.fig_num = max(existing_figures) + 1
            else:
                self.fig_num = 1

    def reset(self):
        self.metrics = defaultdict(
            lambda: defaultdict(lambda: {"epochs": [], "avgs": []})
        )
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )
        self.y_axes = {}
        self.last_epoch = -1
        self.y_limits = {}

        if self.show:
            self.out = Output()
            display.display(self.out)

    def end_loop(self, epoch: int):
        self.last_epoch = epoch
        for metric_name, metric_dict in self.metrics_loop.items():
            for category_name, category_dict in metric_dict.items():
                metric = self.metrics[metric_name][category_name]
                metric["epochs"].append(epoch)
                metric["avgs"].append(category_dict["avg"])
        self.metrics_loop = defaultdict(
            lambda: defaultdict(lambda: {"val": 0.0, "count": 0, "avg": 0.0})
        )

    def update(
        self,
        category_name: str,
        metric_name: str,
        val: float,
        count: int = 1,
        y_axis: str | None = None,
        y_limits: tuple[float | None, float | None] = (None, None),
    ):
        if y_axis is None:
            y_axis = metric_name
        self.y_axes[metric_name] = y_axis
        self.y_limits[metric_name] = y_limits

        metric = self.metrics_loop[metric_name][category_name]

        metric["val"] += val
        metric["count"] += count
        metric["avg"] = metric["val"] / metric["count"]

    def get_last(self, category_name: str, metric_name: str):
        return self.metrics_loop[metric_name][category_name]["avg"]

    def save_metrics(self, save_path: str) -> None:
        with open(save_path, "w") as fp:
            json.dump(self.metrics, fp, sort_keys=True)

    def visualise(
        self,
        intervals: List[Tuple[int, int]] = [(0, 0)],
        save_paths: Optional[Sequence[Path | None]] = None,
    ):
        # Inspired from https://gitlab.com/robindar/dl-scaman_checker/-/blob/main/src/dl_scaman_checker/TP01.py
        if self.show is False and (
            save_paths is None or all(path is None for path in save_paths)
        ):
            return

        if save_paths is None:
            save_paths = [None] * len(intervals)

        if len(save_paths) != len(intervals):
            raise ValueError("intervals and save_paths should have the same length.")

        metrics_index: Dict[str, int] = {}
        categories_index: Dict[str, int] = {}
        for i, (metric_name, metric_dict) in enumerate(self.metrics.items()):
            metrics_index[metric_name] = i
            for category_name in metric_dict.keys():
                if category_name not in categories_index.keys():
                    categories_index[category_name] = len(categories_index)

        n_metrics = len(metrics_index)
        scale = max(ceil(n_metrics**0.5), 1)
        nrows = scale
        ncols = max((n_metrics + scale - 1) // scale, 1)
        cmap = plt.get_cmap("tab10")

        categories_colors = {
            label: cmap(i) for i, label in enumerate(categories_index.keys())
        }
        legend_space = 0.5
        figsize = (7 * ncols, 3.5 * nrows + legend_space)
        legend_y_position = legend_space / figsize[1]
        ax0 = None

        for interval, save_path in zip(intervals, save_paths):
            start = (
                interval[0]
                if interval[0] >= 0
                else max(0, self.last_epoch + interval[0])
            )
            end = (
                self.last_epoch + interval[1]
                if interval[1] <= 0
                else min(interval[1], self.last_epoch)
            )
            x_margin = (end - start) * 0.05
            x_extent_min = start - x_margin
            x_extent_max = end + x_margin

            self._create_fig_num()
            fig = plt.figure(self.fig_num, figsize=figsize)
            plt.clf()

            for metric_name, metric_dict in self.metrics.items():
                # Extract the values in the interval
                cropped_metric_dict = {}
                y_extent_min = np.inf
                y_extent_max = -np.inf
                for category_name, category_dict in metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    index_start = bisect.bisect_left(epochs, start)
                    index_end = bisect.bisect_right(epochs, end) - 1
                    index_start_values = bisect.bisect_left(epochs, x_extent_min)
                    if index_start_values > 0:
                        index_start_values -= 1
                    index_end_values = bisect.bisect_right(epochs, x_extent_max) - 1
                    if index_end_values < len(epochs) - 1:
                        index_end_values += 1

                    cropped_metric_dict[category_name] = {}
                    cropped_metric_dict[category_name]["epochs"] = epochs[
                        index_start_values : index_end_values + 1
                    ]
                    cropped_metric_dict[category_name]["avgs"] = values[
                        index_start_values : index_end_values + 1
                    ]

                    if index_end > index_start:
                        y_extent_min = min(
                            y_extent_min, min(values[index_start : index_end + 1])
                        )
                        y_extent_max = max(
                            y_extent_max, max(values[index_start : index_end + 1])
                        )

                # Plot the metrics
                index = metrics_index[metric_name]
                ax = fig.add_subplot(nrows, ncols, index + 1, sharex=ax0)
                if ax0 is None:
                    ax0 = ax
                for category_name, category_dict in cropped_metric_dict.items():
                    epochs = category_dict["epochs"]
                    values = category_dict["avgs"]

                    # Remove the epochs before from_epoch
                    epochs_length = max(epochs) - min(epochs) if len(epochs) > 0 else 0

                    fmt = "-" if epochs_length > 15 else "-o"
                    ax.plot(
                        epochs,
                        values,
                        fmt,
                        color=categories_colors[category_name],
                        # label=category_name,
                    )

                ax.grid(alpha=0.5)
                y_lower = y_extent_min
                y_upper = y_extent_max

                if self.y_limits[metric_name][0] is not None:
                    y_lower = self.y_limits[metric_name][0]
                if self.y_limits[metric_name][1] is not None:
                    y_upper = self.y_limits[metric_name][1]

                if y_lower != y_upper and np.isfinite(y_lower) and np.isfinite(y_upper):
                    y_margin = (y_upper - y_lower) * 0.05
                    y_lower -= y_margin
                    y_upper += y_margin

                    ax.set_ylim(y_lower, y_upper)

                if index >= n_metrics - ncols:
                    ax.set_xlabel("Epoch")
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                    ax.set_xlim(start - x_margin, end + x_margin)
                else:
                    ax.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                ax.set_ylabel(self.y_axes[metric_name])
                ax.set_title(f"{metric_name}")

            lines = [
                mlines.Line2D([], [], color=color, linestyle="-", label=label)
                for label, color in categories_colors.items()
            ]

            if len(lines) > 0:
                fig.legend(
                    handles=lines,
                    loc="upper center",
                    bbox_to_anchor=(0.5, legend_y_position),
                    ncol=len(lines),
                )

            fig.tight_layout(rect=(0.0, legend_y_position, 1.0, 1.0))

            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=200)

        if self.show:
            with self.out:
                plt.show()
                display.clear_output(wait=True)
