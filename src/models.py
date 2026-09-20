from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "DEATH_EVENT"

def train_models(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "lda": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "perceptron": Pipeline([("scaler", StandardScaler()), ("model", Perceptron(max_iter=2000, random_state=RANDOM_STATE))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        if hasattr(model, "predict_proba"):
            score_values = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            score_values = model.decision_function(X_test)
        else:
            score_values = pred

        rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, pred),
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, score_values),
        })

        pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(output_dir / f"{name}_classification_report.csv")
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot()
        plt.title(f"{name.replace('_',' ').title()} — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, score_values)
        plt.title(f"{name.replace('_',' ').title()} — ROC Curve")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_roc.png", dpi=150)
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, score_values)
        plt.title(f"{name.replace('_',' ').title()} — Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_precision_recall.png", dpi=150)
        plt.close()

    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(output_dir / "model_comparison.csv", index=False)
