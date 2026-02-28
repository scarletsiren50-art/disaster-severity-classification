from models.cnn_optimized import run_cnn
import matplotlib.pyplot as plt

if __name__ == "__main__":

    history = run_cnn()

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("CNN Training Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.savefig("results/graphs/week6_cnn_accuracy.png")
    plt.show()