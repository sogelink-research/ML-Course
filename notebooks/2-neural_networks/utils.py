import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision import transforms
from tqdm.notebook import tqdm


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
