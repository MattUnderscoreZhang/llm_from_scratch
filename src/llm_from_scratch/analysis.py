import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def plot_loss_history(loss_history: dict[str, list[float]], filename: str) -> None:
    plt.plot(loss_history["training"], color="orange")
    plt.plot(loss_history["validation"], color="blue")
    plt.title("Loss history")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.savefig(filename)
