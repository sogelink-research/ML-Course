# Installation

Use one of the following commands depending on your GPU situation:

- No GPU:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

- CUDA 12.6:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

- CUDA 11.8:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

Then install the rest of the requirements:

```bash
pip3 install -r requirements.txt
```
