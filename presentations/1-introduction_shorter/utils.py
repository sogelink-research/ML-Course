from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


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


def create_meshgrid(X: NDArray) -> tuple[NDArray, NDArray]:
    # Create a meshgrid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    return xx, yy


def plot_decision_boundaries(
    model,
    meshgrid: tuple[NDArray, NDArray],
    X_test: NDArray,
    y_test: NDArray,
    class_names: list[str],
    ax: Axes | None = None,
):
    xx, yy = meshgrid

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


class IrisData:
    def __init__(self, features_used: tuple[int, int]):
        self.features_used = features_used
        self.data = load_iris()
        self.feature_names = [self.data.feature_names[i] for i in features_used]
        self.target_names = self.data.target_names
        self.X = self.data.data[:, features_used]
        self.y = self.data.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=2
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X = scaler.transform(self.X)

    def get_data(self) -> Bunch:
        return self.data


class ModelEvaluation:
    def __init__(self, model, iris_data: IrisData):
        self.model = model
        self.iris_data = iris_data
        self.fit_predict_evaluate()

    def fit_predict_evaluate(self):
        self.model.fit(self.iris_data.X_train, self.iris_data.y_train)

    def plot_decision_boundaries_all(self, figsize: tuple[int, int]):
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        plt.suptitle(
            f"{self.model.__class__.__name__}",
            fontsize="xx-large",
            y=1.02,
        )
        meshgrid = create_meshgrid(self.iris_data.X)
        plot_decision_boundaries(
            self.model,
            meshgrid,
            self.iris_data.X_train,
            self.iris_data.y_train,
            self.iris_data.feature_names,
            ax=axs[0],
        )
        axs[0].set_title(axs[0].get_title() + " (Training Data)")
        plot_decision_boundaries(
            self.model,
            meshgrid,
            self.iris_data.X_test,
            self.iris_data.y_test,
            self.iris_data.feature_names,
            ax=axs[1],
        )
        axs[1].set_title(axs[1].get_title() + " (Testing Data)")


# def fit_predict_evaluate_iris(
#     model, data: Bunch, features_used: tuple[int, int], figsize: tuple[int, int]
# ):
#     X, y, feature_names, class_names = (
#         data.data[:, features_used],
#         data.target,
#         [data.feature_names[i] for i in features_used],
#         data.target_names,
#     )

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=0
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model.fit(X_train, y_train)
#     # y_pred = model.predict(X_test)

#     # accuracy = accuracy_score(y_test, y_pred)
#     # print(f"Accuracy: {accuracy:.4f}")
#     # fig, axs = plt.subplots(1, 4, figsize=(24, 5))
#     fig, axs = plt.subplots(1, 2, figsize=figsize)
#     plt.suptitle(
#         f"{model.__class__.__name__} using features={features_used}",
#         fontsize="xx-large",
#         y=1.02,
#     )

#     # plot_confusion_matrix(y_test, y_pred, class_names, ax=axs[0])
#     # axs[0].set_title(axs[0].get_title() + " (Testing Data)")
#     # plot_evaluation(y_test, y_pred, class_names, ax=axs[1])
#     # axs[1].set_title(axs[1].get_title() + " (Testing Data)")
#     # plot_decision_boundaries(model, X_train, y_train, feature_names, ax=axs[2])
#     # axs[2].set_title(axs[2].get_title() + " (Training Data)")
#     # plot_decision_boundaries(model, X_test, y_test, feature_names, ax=axs[3])
#     # axs[3].set_title(axs[3].get_title() + " (Testing Data)")

#     plot_decision_boundaries(model, X_train, y_train, feature_names, ax=axs[0])
#     axs[0].set_title(axs[0].get_title() + " (Training Data)")
#     plot_decision_boundaries(model, X_test, y_test, feature_names, ax=axs[1])
#     axs[1].set_title(axs[1].get_title() + " (Testing Data)")

#     return model
