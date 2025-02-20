import torch
import torch.nn as nn
import tqdm


class SegmentationConvolutionalNetwork(nn.Module):

    def __init__(self, image_size: tuple[int, int]):
        super().__init__()

        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3)
        self.linear = nn.LazyLinear(out_features=image_size[0] * image_size[1])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = image.transpose(1, -1)
        output = nn.functional.relu(
            nn.functional.max_pool2d(self.conv1(output), kernel_size=3)
        )
        output = nn.functional.relu(
            nn.functional.max_pool2d(self.conv2(output), kernel_size=3)
        )
        return nn.functional.relu(self.linear(output.view(image.shape[0], -1)))

    def training_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        logits = self.forward(images)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.view(targets.shape[0], -1).float()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.nanmean().item()

    def validation_step(self, images: torch.Tensor, targets: torch.Tensor) -> float:
        logits = self.forward(images)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.view(targets.shape[0], -1).float()
        )

        return loss.nanmean().item()

    def run(self, dataloaders: tuple[torch.utils.data.DataLoader], epochs: int):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        progress = tqdm.tqdm(range(epochs))

        for epoch in progress:
            losses = dict(training=[], validation=[])

            self.train()

            for images, targets in dataloaders[0]:
                losses["training"].append(
                    self.training_step(images, targets, optimizer)
                )

            with torch.no_grad():
                self.eval()

                for images, targets in dataloaders[1]:
                    losses["validation"].append(self.validation_step(images, targets))

            progress.set_description(
                "[training.loss] {:.4f}, [validation.loss] {:.4f}".format(
                    sum(losses["training"]) / len(losses["training"]),
                    sum(losses["validation"]) / len(losses["validation"]),
                ),
                refresh=True,
            )
