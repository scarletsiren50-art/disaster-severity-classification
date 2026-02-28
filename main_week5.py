from models.baseline_ml import run_baseline_models

if __name__ == "__main__":
    results = run_baseline_models()

    print("\nFinal Accuracy Summary:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")