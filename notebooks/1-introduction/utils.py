from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


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
