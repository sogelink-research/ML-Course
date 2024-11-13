# Notebooks

## Purpose

This directory contains Jupyter notebooks with the same organization as the [`presentations`](../presentations) directory. The notebooks are intended to provide an environment and tips to explore some of the techniques mentioned in the presentations.

## Organization

The structure of this directory is the following:

```bash
notebooks
├── README.md
├── .venv/
├── shared/
│   └── requirements.txt
├── 1-introduction/
│   ├── Demo-Classification_sklearn.ipynb
│   ├── Exercise-Classification_sklearn.ipynb
│   └── Solution-Classification_sklearn.ipynb
├── 2-neural_networks/
│   ├── Demo-[...].ipynb
│   ├── Exercise-[...].ipynb
│   ├── Solution-[...].ipynb
│   └── ...
└── ...
```

A few comments about the structure:

- The `shared` directory contains a `requirements.txt` file with the Python packages needed to run the notebooks. This file is used to create a virtual environment as explained in the next section [`Setup`](#setup).
- There are two types of notebooks: `Demo` and `Exercise`. The `Demo` notebooks contain examples and explanations and can be run without any modification. The `Exercise` notebooks contain exercises that require you to solve them, and have a `Solution` version containing one solution.

## Setup

To run the notebooks, you need to create a virtual environment and install the required packages. The following sections explain how to do it from the command line:

### macOS and Linux

```bash
cd notebooks
python3 -m venv .venv
source .venv/bin/activate
pip install -r shared/requirements.txt
```

### Windows

```bash
cd notebooks
python3 -m venv .venv
.venv\Scripts\activate
pip install -r shared\requirements.txt
```
