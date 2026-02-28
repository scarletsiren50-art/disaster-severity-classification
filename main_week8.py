from models.resnet_model import run_resnet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == "__main__":

    history, model, val_generator = run_resnet()

    # ---- Accuracy Plot ----
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("ResNet Training Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.savefig("results/graphs/week8_resnet_accuracy.png")
    plt.show()

    # ---- Confusion Matrix ----
    val_generator.reset()

    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=val_generator.class_indices.keys(),
                yticklabels=val_generator.class_indices.keys())

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("ResNet50 Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/graphs/resnet_confusion_matrix.png")
    plt.show()