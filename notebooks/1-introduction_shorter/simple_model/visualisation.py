import matplotlib.pyplot as plt
from IPython.display import clear_output, display


class TrainingVisualisation:

    def __init__(self):
        self.training_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.keep_updating = True

        # Create the figure once and update it dynamically
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax_losses = plt.subplots(figsize=(10, 5))

        # Twin axis for learning rate
        self.ax_lr = self.ax_losses.twinx()

        # Connect to close event
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def on_close(self, event):
        """Callback to detect when the figure window is closed."""
        print("Plot window closed. Stopping updates.")
        self.keep_updating = False

    def update_plots(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Update the plot with new data."""
        if not self.keep_updating:
            return

        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        self.learning_rates.append(lr)

        # Clear the plot
        self.ax_losses.clear()
        self.ax_lr.clear()

        # Plot Losses
        self.ax_losses.plot(self.training_losses, label="Train Loss", color="blue")
        self.ax_losses.plot(
            self.validation_losses, label="Validation Loss", color="red"
        )
        self.ax_losses.set_xlabel("Epoch")
        self.ax_losses.set_ylabel("Loss")
        self.ax_losses.legend(loc="upper left")
        self.ax_losses.grid(True)

        # Plot Learning Rate
        self.ax_lr.plot(
            self.learning_rates,
            label="Learning Rate",
            color="green",
            linestyle="dashed",
        )
        self.ax_lr.set_ylabel("Learning Rate")
        self.ax_lr.legend(loc="upper right")

        plt.title(f"Epoch {epoch}")
        plt.tight_layout()

        clear_output(wait=True)  # Clears previous output
        display(self.fig)  # Displays updated figure in Jupyter

        plt.draw()  # Update the plot
        # plt.pause(0.1)  # Allow time for rendering

    # def close(self):
    #     plt.ioff()  # Turn off interactive mode when training is done
    #     plt.show()  # Final show of the plot
