import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler

def load_data():
    try:
        digits = load_digits()
        X, y = digits.data, digits.target
        print("Dataset caricato da sklearn.datasets.load_digits()")
    except ImportError:
        df = pd.read_csv("digits.csv")
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        print("load_digits() fallito, dataset caricato da digits.csv")
    return X, y

def visualize_pca(X_pca, y):
    X_mirrored = X_pca.copy()
    X_mirrored[:, 0] *= -1

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_mirrored[:, 0],
        X_mirrored[:, 1],
        c=y,
        cmap='tab10',
        alpha=0.7,
        s=20
    )
    plt.legend(
        *scatter.legend_elements(),
        title="Digits",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.title("PCA")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.tight_layout()
    plt.show()
    return X_mirrored

# Caricamento dati 
X, y = load_data()

# Scree Plot - Varianza spiegata da ciascuna componente PCA
pca_full = PCA()
pca_full.fit(X)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel("Numero di componenti")
plt.ylabel("Varianza cumulativa spiegata")
plt.title("Scree Plot - PCA")
plt.grid(True)
plt.show()

# Proiezione PCA 2D per visualizzazione
pca_viz = PCA(n_components=2, svd_solver='randomized', random_state=42)
X_pca = pca_viz.fit_transform(X)
visualize_pca(X_pca, y)

# Suddivisione train/test e pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, svd_solver='randomized', random_state=42)),
    ('svm', SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42))
])

# Addestramento, predizione e metriche
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuratezza SVM su PCA (pipeline): {acc:.4f}\n")
print(classification_report(y_test, y_pred))

