# Notes

## Evaluation

Bounding boxes to use for the evaluation:

```python
minx_miny_maxx_maxy = [
    (94000, 430000, 95000, 431000),
    (70000, 417000, 71000, 418000),
    (73000, 406000, 74000, 407000),
    (33000, 391000, 34000, 392000),
    (146000, 376000, 147000, 377000),
    (181000, 391000, 182000, 392000),
    (185000, 367000, 186000, 368000),
    (198000, 320000, 199000, 321000),
    (130000, 540000, 131000, 541000),
    (102000, 498000, 103000, 499000),
    (120000, 486000, 121000, 487000),
    (174000, 560000, 175000, 561000),
    (181000, 591000, 182000, 592000),
    (182000, 533000, 183000, 534000),
    (258000, 533000, 259000, 534000),
    (237000, 511000, 238000, 512000),
    (226000, 502000, 227000, 503000),
    (242000, 485000, 243000, 486000),
    (211000, 456000, 212000, 457000),
    (72000, 446000, 73000, 447000),
    (69000, 440000, 70000, 441000),
    (81000, 455000, 82000, 456000),
    (88000, 436000, 89000, 437000),
    (136000, 455000, 137000, 456000),
    (128000, 425000, 129000, 426000),
]
```

Link to download the data for the evaluation: <https://drive.proton.me/urls/JZWBH7WR68#uEZjHBCoa6Ui>.

## Parameters

Possible parameters for the model:

```python
# ------------------------------ Data selection ------------------------------ #
minx_miny_maxx_maxy = [
    (120000, 482000, 125000, 487000),
    (157000, 475000, 162000, 480000),
]  # Coordinates of the areas of interest (minx, maxy, maxx, miny)
filter_buildings = True  # True to filter small buildings

# ----------------------------- Data preparation ----------------------------- #
image_size = 512  # Size of the images processed by the model
nodata = -10  # Value to replace nodata with

# ----------------------------- Model parameters ----------------------------- #
encoder_channels = [16, 32, 64]  # Number of channels in the encoder and decoder
layers_downsample = 2  # Number of convolutional layers in the encoder
layers_upsample = 2  # Number of convolutional layers in the decoder

# ---------------------------- Training parameters --------------------------- #
batch_size = 8  # Size of the batches (number of images processed at once)
train_proportion = 0.8  # Proportion of the data used for training

optimizer_type = "adam"  # Type of optimizer to use ("adam" or "sgd")
initial_learning_rate = 0.01  # Initial learning rate for the optimizer
max_epochs = 200  # Maximum number of iterations to train the model
stop_early_after = (
    10  # Number of epochs without improvement before stopping the training
)
```

## Default Parameters

```python
# ------------------------------ Data selection ------------------------------ #
minx_miny_maxx_maxy = [
    (None, None, None, None),
    (None, None, None, None),
    ...,
]  # Coordinates of the areas of interest (minx, maxy, maxx, miny)
filter_buildings = True  # True to filter small buildings

# ----------------------------- Data preparation ----------------------------- #
image_size = None  # Size of the images processed by the model
nodata = None  # Value to replace nodata with

# ----------------------------- Model parameters ----------------------------- #
encoder_channels = [None, None, ...]  # Number of channels in the encoder and decoder
layers_downsample = None  # Number of convolutional layers in the encoder
layers_upsample = None  # Number of convolutional layers in the decoder

# ---------------------------- Training parameters --------------------------- #
batch_size = None  # Size of the batches (number of images processed at once)
train_proportion = None  # Proportion of the data used for training

optimizer_type = None  # Type of optimizer to use ("adam" or "sgd")
initial_learning_rate = None  # Initial learning rate for the optimizer
max_epochs = None  # Maximum number of iterations to train the model
stop_early_after = (
    None  # Number of epochs without improvement before stopping the training
)
```
