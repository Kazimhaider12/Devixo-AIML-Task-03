"""
Parts 2, 3 & 4: End-to-End Model Training, Cross-Validation, Hyperparameter Tuning,
Feature Importance Analysis, Visualization, and Serialization (Joblib + Pickle).
"""

import json
import os
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import (
    build_preprocessor,
    engineer_features,
    load_and_clean_data,
)

warnings.filterwarnings("ignore")

# Dark-themed sleek plotting style
plt.rcParams.update({
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#E2E8F0",
    "text.color": "#F8FAFC",
    "xtick.color": "#CBD5E1",
    "ytick.color": "#CBD5E1",
    "grid.color": "#334155",
    "font.size": 10,
})

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = "models"
PLOTS_DIR = os.path.join("outputs", "plots")
METRICS_PATH = os.path.join("outputs", "metrics_summary.json")


def get_candidate_models():
    """Return 5 diverse ML models (plus optional XGBoost if installed)."""
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            eval_metric="logloss", random_state=42
        )
    except ImportError:
        pass
    return models


def plot_model_comparison(results_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics_to_plot = ["CV ROC-AUC", "Test ROC-AUC", "Test F1", "Test Accuracy"]
    x = np.arange(len(results_df))
    width = 0.2
    colors = ["#38BDF8", "#818CF8", "#34D399", "#F472B6"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        bars = ax.bar(x + (i - 1.5) * width, results_df[metric], width, label=metric, color=color, edgecolor="none")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color="#F8FAFC")

    ax.set_title("Baseline Model Comparison (5-Fold CV & Test Set)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_roc_curves(roc_data: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    palette = ["#38BDF8", "#818CF8", "#34D399", "#F472B6", "#FBBF24", "#FB7185"]
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), palette):
        lw = 2.6 if "Tuned" in name else 1.8
        ls = "-" if "Tuned" in name else "--"
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=lw, linestyle=ls)

    ax.plot([0, 1], [0, 1], color="#64748B", linestyle=":", linewidth=1.2, label="Random Baseline (AUC = 0.50)")
    ax.set_title("ROC Curves - Customer Churn Classification", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No Churn (0)", "Churn (1)"],
        yticklabels=["No Churn (0)", "Churn (1)"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, save_path: str):
    top_df = fi_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = plt.cm.cool(np.linspace(0.25, 0.85, len(top_df)))
    ax.barh(top_df["Feature"], top_df["Importance"], color=colors)
    ax.set_title("Top 15 Feature Importances (Tuned Best Model)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # PART 1: Data Preparation
    df, cleaning_stats = load_and_clean_data(DATA_PATH)
    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        threshold="0.35*mean"
    )

    # PART 2: Model Development (5 Models + 5-Fold Stratified CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_candidate_models()
    baseline_results = []
    roc_data = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", feature_selector),
            ("classifier", clf),
        ])

        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "f1", "accuracy", "precision", "recall"],
            n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        test_auc = roc_auc_score(y_test, y_prob)
        roc_data[name] = (fpr, tpr, test_auc)

        baseline_results.append({
            "Model": name,
            "CV ROC-AUC": round(float(cv_scores["test_roc_auc"].mean()), 4),
            "CV ROC-AUC Std": round(float(cv_scores["test_roc_auc"].std()), 4),
            "CV F1": round(float(cv_scores["test_f1"].mean()), 4),
            "Test Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Test Precision": round(float(precision_score(y_test, y_pred)), 4),
            "Test Recall": round(float(recall_score(y_test, y_pred)), 4),
            "Test F1": round(float(f1_score(y_test, y_pred)), 4),
            "Test ROC-AUC": round(float(test_auc), 4),
        })

    results_df = pd.DataFrame(baseline_results).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    print("\n=== Baseline Model Comparison ===")
    print(results_df.to_string(index=False))

    # PART 3: Hyperparameter Tuning & Feature Importance
    gb_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ])
    gb_param_grid = {
        "classifier__n_estimators": [120, 200],
        "classifier__learning_rate": [0.05, 0.08],
        "classifier__max_depth": [3, 4],
        "classifier__subsample": [0.85, 1.0],
    }

    grid_search = GridSearchCV(
        gb_pipe,
        param_grid=gb_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    y_pred_best = best_pipeline.predict(X_test)
    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_best)
    best_test_auc = roc_auc_score(y_test, y_prob_best)
    roc_data["Gradient Boosting (Tuned)"] = (fpr_b, tpr_b, best_test_auc)

    tuned_metrics = {
        "Model": "Gradient Boosting (Tuned)",
        "Best Params": grid_search.best_params_,
        "Best CV ROC-AUC": round(float(grid_search.best_score_), 4),
        "Test Accuracy": round(float(accuracy_score(y_test, y_pred_best)), 4),
        "Test Precision": round(float(precision_score(y_test, y_pred_best)), 4),
        "Test Recall": round(float(recall_score(y_test, y_pred_best)), 4),
        "Test F1": round(float(f1_score(y_test, y_pred_best)), 4),
        "Test ROC-AUC": round(float(best_test_auc), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_best).tolist(),
    }

    print("\n=== Tuned Best Model Results ===")
    print(json.dumps(tuned_metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

    pre_fitted = best_pipeline.named_steps["preprocessor"]
    sel_fitted = best_pipeline.named_steps["feature_selection"]
    clf_fitted = best_pipeline.named_steps["classifier"]

    all_feature_names = pre_fitted.get_feature_names_out()
    selected_mask = sel_fitted.get_support()
    selected_features = [f.replace("num__", "").replace("cat__", "") for f in all_feature_names[selected_mask]]

    fi_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": clf_fitted.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\nFeature Selection kept {len(selected_features)} / {len(all_feature_names)} encoded features.")
    print("Top 10 Features:")
    print(fi_df.head(10).to_string(index=False))

    plot_model_comparison(results_df, os.path.join(PLOTS_DIR, "model_comparison.png"))
    plot_roc_curves(roc_data, os.path.join(PLOTS_DIR, "roc_curves.png"))
    plot_confusion_matrix(confusion_matrix(y_test, y_pred_best), "Tuned Gradient Boosting", os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_feature_importance(fi_df, os.path.join(PLOTS_DIR, "feature_importance.png"))

    # PART 4: Save the Best Model (Joblib + Pickle) & Verify Reload
    joblib_path = os.path.join(MODELS_DIR, "best_model.joblib")
    pickle_path = os.path.join(MODELS_DIR, "best_model.pkl")

    joblib.dump(best_pipeline, joblib_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(best_pipeline, f)

    loaded_joblib = joblib.load(joblib_path)
    with open(pickle_path, "rb") as f:
        loaded_pickle = pickle.load(f)

    sample_X = X_test.iloc[:5]
    pred_j = loaded_joblib.predict_proba(sample_X)[:, 1]
    pred_p = loaded_pickle.predict_proba(sample_X)[:, 1]
    assert np.allclose(pred_j, pred_p), "Joblib and Pickle predictions differ!"
    print(f"\nSaved & verified best model: {joblib_path} and {pickle_path}")

    summary = {
        "dataset_stats": cleaning_stats,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "churn_rate_percent": round(float(y.mean() * 100), 2),
        "total_encoded_features": int(len(all_feature_names)),
        "selected_features_count": int(len(selected_features)),
        "baseline_models": baseline_results,
        "tuned_best_model": tuned_metrics,
        "top_10_features": fi_df.head(10).to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {METRICS_PATH}")


if __name__ == "__main__":
    main()
