import matplotlib.pyplot as plt

models = ["Logistic Regression", "SVM", "Random Forest"]
accuracy = [0.4875, 0.5917, 0.6333]

plt.figure()
plt.bar(models, accuracy)
plt.ylabel("Accuracy")
plt.title("Week 5: Baseline Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/graphs/week5_comparison.png")
plt.show()