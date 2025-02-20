import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from models.nn import SegmentationConvolutionalNetwork

if __name__ == "__main__":
    model = SegmentationConvolutionalNetwork(image_size=(32, 32))
    dataloaders = (
        DataLoader(
            TensorDataset(
                torch.randn(size=(8192, 32, 32, 1)),
                torch.randint(low=0, high=1, size=(8192, 32, 32, 1)),
            ),
            batch_size=16,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(
                torch.randn(size=(2048, 32, 32, 1)),
                torch.randint(low=0, high=1, size=(2048, 32, 32, 1)),
            ),
            batch_size=16,
            shuffle=False,
        ),
    )

    model.run(dataloaders=dataloaders, epochs=25)
