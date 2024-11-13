# Presentations

## Purpose

This directory contains HTML slides with the same organization as the [`notebooks`](../notebooks) directory. The slides can be opened with any web browser.

## Organization

The structure of this directory is the following:

```bash
presentations
├── README.md
├── .venv/
├── _quarto.yml
├── .gitignore
├── shared/
│   ├── requirements.txt
│   └── ...
├── slides/
│   ├── 1-introduction
│   │   ├── slides.html
│   │   └── website.html
│   ├── 2-neural_networks
│   │   ├── slides.html
│   │   └── website.html
│   └── ...
├── 1-introduction/
│   ├── images/
│   ├── references.bib
│   └── slides.qmd
├── 2-neural_networks/
│   ├── images/
│   ├── references.bib
│   └── slides.qmd
└── ...

```

A few comments about the structure:

- The `shared` directory contains a `requirements.txt` file with the Python packages needed to render the slides using Quarto. This file is used to create a virtual environment as explained in the next section [`Render slides`](#render-slides).
- The `shared` directory also contains a few files used to customize the slides.
- Each presentation has two directories: one with the source files (at the level just below `presentations`) and one with the final slides (inside [`slides`](./slides/)). The final slides are standalone HTML documents that can be opened with any web browser, called `slides.html`. There is also a linear version exported as a website called `website.html`, with the same content in a different format.

## Render slides

The slides are rendered using [Quarto](https://quarto.org/). The Quarto version used to render the slides is [`v1.6.32`](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.6.32).

To juggle between Quarto versions, I use [qvm](https://github.com/dpastoor/qvm), a Quarto Version Manager. You can see [here](https://github.com/dpastoor/qvm/releases) how to install it on MacOS or Linux. I installed on Ubuntu using:

```bash
sudo wget https://github.com/dpastoor/qvm/releases/download/v0.3.0/qvm_Linux_x86_64.tar.gz -O /tmp/qvm.tar.gz
sudo tar xzf /tmp/qvm.tar.gz qvm
sudo mv qvm /usr/local/bin/qvm
sudo chmod +x /usr/local/bin/qvm
```

Once Quarto is installed, you should add the [Reveal-header Extension For Quarto](https://github.com/shafayetShafee/reveal-header):

```bash
cd presentations
quarto add shafayetShafee/reveal-header
```

Then, you just need to activate the Python virtual environment created from [`shared/requirements.txt`](./shared/requirements.txt), and then use the following to render the slides and the website version:

```bash
quarto render 1-introduction/slides.qmd --to revealjs       # Slides
quarto render 1-introduction/slides.qmd --to html           # Website
```
