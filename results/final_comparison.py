import matplotlib.pyplot as plt
import pandas as pd
import os

# Final validation accuracies
models = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "CNN (Optimized)",
    "ResNet50 (Frozen)"
]

accuracies = [
    0.4875,
    0.5917,
    0.6333,
    0.5125,
    0.5583
]

# Create DataFrame
df = pd.DataFrame({
    "Model": models,
    "Validation Accuracy": accuracies
})

# Save CSV
os.makedirs("results/reports", exist_ok=True)
df.to_csv("results/reports/final_model_comparison.csv", index=False)

# Plot graph
plt.figure()
plt.bar(models, accuracies)
plt.xticks(rotation=25)
plt.ylabel("Validation Accuracy")
plt.title("Final Model Comparison (Week 5–8)")
plt.tight_layout()

os.makedirs("results/graphs", exist_ok=True)
plt.savefig("results/graphs/final_model_comparison.png")

plt.show()

print("\nFinal Comparison Table:\n")
print(df)