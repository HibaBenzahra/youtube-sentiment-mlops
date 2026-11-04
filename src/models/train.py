import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"
METRICS_PATH = "models/metrics.json"
CM_PATH = "models/confusion_matrix.png"
COMPARISON_PATH = "models/comparison.png"


def load_data():
    train = pd.read_csv(TRAIN_PATH).dropna()
    test = pd.read_csv(TEST_PATH).dropna()
    return train, test


def build_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def compare_models(X_train, y_train, X_test, y_test):
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro"),
        }
        print(f"{name} -> accuracy={results[name]['accuracy']:.3f} | f1_macro={results[name]['f1_macro']:.3f}")
    return results


def optimize_logreg(X_train, y_train):
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        param_grid={"C": [0.5, 1, 5, 10, 20]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur F1 (validation croisée) :", round(grid.best_score_, 3))
    return grid.best_estimator_


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    print("\nAccuracy :", round(accuracy_score(y_test, pred), 3))
    print(classification_report(y_test, pred, target_names=["Négatif", "Neutre", "Positif"]))
    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred, labels=[-1, 0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Négatif", "Neutre", "Positif"],
                yticklabels=["Négatif", "Neutre", "Positif"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    return report


def measure_inference_time(model, vectorizer, texts):
    batch = texts[:50]
    start = time.perf_counter()
    model.predict(vectorizer.transform(batch))
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"Temps d'inférence pour 50 commentaires : {elapsed_ms:.1f} ms")
    return elapsed_ms


def plot_comparison(results):
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    x = range(len(names))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], acc, width=0.4, label="Accuracy")
    plt.bar([i + 0.2 for i in x], f1, width=0.4, label="F1 macro")
    plt.xticks(list(x), names)
    plt.ylim(0, 1)
    plt.title("Comparaison des modèles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    train, test = load_data()
    vectorizer, X_train = build_vectorizer(train["text"])
    X_test = vectorizer.transform(test["text"])
    y_train, y_test = train["label"], test["label"]

    print("=== Comparaison des algorithmes ===")
    results = compare_models(X_train, y_train, X_test, y_test)
    plot_comparison(results)

    print("\n=== Optimisation de la Logistic Regression ===")
    best_model = optimize_logreg(X_train, y_train)

    print("\n=== Évaluation finale ===")
    report = evaluate(best_model, X_test, y_test)
    inference_ms = measure_inference_time(best_model, vectorizer, test["text"].tolist())

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"comparison": results, "report": report, "inference_ms_50": inference_ms}, f, indent=2)
    print("\nModèle et vectoriseur sauvegardés dans models/")
