"""PCA and SVM classification of the scikit-learn Digits dataset.

All exploration and model selection use the training set only. The test set
is used once, at the end, to evaluate the selected model and the 2-component
reference model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE = 0.2
COMPONENT_OPTIONS = [2, 5, 10, 15, 20, 30, 40, 50, 60]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def build_pipeline(n_components=None):
    """Standardize, optionally reduce with PCA, then classify with an RBF SVM."""
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("svm", SVC(kernel="rbf", gamma="scale", C=1.0)))
    return Pipeline(steps)


def cross_validate_components(X, y, options):
    """Mean and standard deviation of 5-fold CV accuracy for each number of components."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for n_components in options:
        scores = cross_val_score(build_pipeline(n_components), X, y, cv=cv)
        results[n_components] = (scores.mean(), scores.std())
    return results


def plot_explained_variance(X_scaled, path):
    cumulative = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
    n_95 = int(np.argmax(cumulative >= 0.95)) + 1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.axvline(n_95, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"Explained variance (95% reached with {n_95} components)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return n_95


def plot_projection(X_scaled, y, path):
    projected = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=y, cmap="tab10", s=12, alpha=0.7)
    ax.legend(*scatter.legend_elements(), title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Training set projected on the first two components")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cv_results(results, baseline, path):
    components = list(results)
    means = np.array([results[n][0] for n in components])
    stds = np.array([results[n][1] for n in components])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(components, means, yerr=stds, marker="o", capsize=3, label="PCA + SVM")
    ax.axhline(baseline[0], color="gray", linestyle="--", linewidth=1, label="SVM without PCA")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("5-fold CV accuracy (training set)")
    ax.set_title("Accuracy vs. number of components")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, n_components, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Test set confusion matrix ({n_components} components)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(figures_dir=FIGURES_DIR):
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}, test samples: {len(X_test)}, features: {X.shape[1]}")

    # Exploration on the standardized training set, as seen by the classifier.
    X_train_scaled = StandardScaler().fit_transform(X_train)
    n_95 = plot_explained_variance(X_train_scaled, figures_dir / "explained_variance.png")
    plot_projection(X_train_scaled, y_train, figures_dir / "pca_projection.png")
    print(f"Components needed for 95% of the variance: {n_95}")

    # Model selection: cross-validation on the training set only.
    results = cross_validate_components(X_train, y_train, COMPONENT_OPTIONS)
    baseline = cross_validate_components(X_train, y_train, [None])[None]
    plot_cv_results(results, baseline, figures_dir / "cv_accuracy.png")

    print("\n5-fold CV accuracy on the training set")
    for n_components, (mean, std) in results.items():
        print(f"  {n_components:>2} components: {mean:.4f} ± {std:.4f}")
    print(f"  no PCA:        {baseline[0]:.4f} ± {baseline[1]:.4f}")

    best = max(results, key=lambda n: results[n][0])
    print(f"\nSelected number of components: {best}")

    # Final evaluation: the only use of the test set.
    print("\nTest set accuracy")
    final_model = build_pipeline(best).fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    for n_components, predictions in ((best, y_pred),
                                      (2, build_pipeline(2).fit(X_train, y_train).predict(X_test))):
        print(f"  {n_components:>2} components: {accuracy_score(y_test, predictions):.4f}")

    print(f"\nClassification report ({best} components)")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, best, figures_dir / "confusion_matrix.png")
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
