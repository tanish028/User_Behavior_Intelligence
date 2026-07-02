import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
import shap

# 1. LABEL CREATION

def create_churn_label(rfm, recency_threshold=180):
    """
    Creates a binary churn label.

    A customer is considered churned if they haven't purchased
    in more than `recency_threshold` days.

    Args:
        rfm (pd.DataFrame): RFM dataframe with a 'recency' column
        recency_threshold (int): Days of inactivity to define churn

    Returns:
        pd.DataFrame: rfm with a new 'Churned' column (1=churned, 0=active)
    """
    rfm = rfm.copy()
    rfm["Churned"] = (rfm["recency"] > recency_threshold).astype(int)

    total = len(rfm)
    churned = rfm["Churned"].sum()
    print(f"Churn label created (threshold = {recency_threshold} days)")
    print(f"  Churned:     {churned:,} ({churned/total*100:.1f}%)")
    print(f"  Not churned: {total - churned:,} ({(total-churned)/total*100:.1f}%)")

    return rfm

# 2. TRAIN THE MODEL


def train_churn_model(rfm):
    """
    Trains an XGBoost classifier to predict churn using R, F, M scores.

    Features: R-Score, F-Score, M-Score (already computed quintile scores)
    Label: Churned (1) or Not Churned (0)

    Args:
        rfm (pd.DataFrame): RFM dataframe with churn label

    Returns:
        model: Trained XGBClassifier
        X_test, y_test: Held-out test set for evaluation
        feature_names: List of feature names used
    """

    # R-Score is excluded: it's derived directly from recency, which defines the churn label.
    # Including it would cause data leakage (AUC ~0.99 but model learns nothing useful).
    # F and M scores are genuinely independent signals — frequency and spend predict churn
    # without directly encoding it.
    features = ["F-Score", "M-Score"]
    X = rfm[features].astype(float)
    y = rfm["Churned"]

    # 80% train, 20% test — stratify to keep class balance equal in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale_pos_weight handles class imbalance:
    # if 70% are churned and 30% are not, we weight the minority class higher
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=100,       # 100 trees
        max_depth=4,            # each tree can ask up to 4 yes/no questions
        learning_rate=0.1,      # how much each tree corrects the previous one
        scale_pos_weight=scale, # handles class imbalance
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )

    model.fit(X_train, y_train)
    print(f"\nModel trained on {len(X_train):,} customers, tested on {len(X_test):,}")

    return model, X_test, y_test, features



# 3. EVALUATE THE MODEL


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints all key metrics.

    Returns:
        auc (float): ROC-AUC score
        fig (matplotlib.Figure): Figure with confusion matrix + ROC curve
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of churn

    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
    print(f"ROC-AUC Score: {auc:.3f}")
    print("  (0.5 = random guessing | 0.75+ = solid | 0.85+ = very good)")

    # Plot: Confusion Matrix + ROC Curve side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Not Churned", "Churned"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=13)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"XGBoost (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--",
                 label="Random Classifier (AUC = 0.5)")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[1].set_title("ROC Curve", fontsize=13)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend(loc="lower right")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    return auc, fig

# 4. FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plots feature importance — which of R, F, M contributed most to predictions.

    Higher importance = that feature was used more often and more effectively
    by the trees to split the data.
    """

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color=["#4a90d9", "#50b86c", "#e05c5c"]
    )

    # Add value labels on bars
    for bar, val in zip(bars, importance[sorted_idx]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_title("Feature Importance — What Drives Churn?", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, max(importance) + 0.08)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

# 5. PREDICT FOR A SINGLE CUSTOMER

def predict_churn(model, f_score, m_score):
    """
    Predicts churn probability for a single customer given their F and M scores.
    R-Score is excluded to avoid data leakage (recency directly defines the churn label).

    Args:
        model: Trained XGBClassifier
        f_score (int): Frequency score 1-5 (5 = very frequent)
        m_score (int): Monetary score 1-5 (5 = very high spend)

    Returns:
        float: Churn probability between 0 and 1
    """
    X = pd.DataFrame([[f_score, m_score]],
                     columns=["F-Score", "M-Score"])
    prob = model.predict_proba(X)[0][1]
    return prob

# 6. SHAP EXPLAINABILITY

def compute_shap_values(model, X_test):
    """
    Computes SHAP values for the test set.

    SHAP (SHapley Additive exPlanations) fairly distributes each prediction
    among the input features based on their contribution.

    Positive SHAP value = feature pushed churn probability UP
    Negative SHAP value = feature pushed churn probability DOWN

    Args:
        model: Trained XGBClassifier
        X_test (pd.DataFrame): Test features

    Returns:
        explainer: SHAP TreeExplainer object
        shap_values: SHAP values array (shape: n_samples x n_features)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, save_path=None):
    """
    Plots SHAP summary — feature importance across all test customers.

    Each dot is one customer. X-axis = SHAP value (impact on prediction).
    Color = feature value (red = high, blue = low).

    Interpretation:
    - F-Score dots on the right with blue color = low frequency → high churn risk
    - F-Score dots on the left with red color = high frequency → low churn risk
    """
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Summary — Feature Impact on Churn Probability", fontsize=13, pad=15)
    plt.tight_layout()

    fig = plt.gcf()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _sigmoid(x):
    """Sigmoid (logistic) function — converts log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-x))


def plot_shap_waterfall(explainer, shap_values, X_test, customer_idx=0, save_path=None):
    """
    Plots a SHAP waterfall for a single customer — shows exactly why
    the model predicted that specific churn probability.

    Starts from the base value (average prediction across all customers),
    then adds each feature's contribution to arrive at the final prediction.

    Args:
        customer_idx (int): Index of customer in X_test to explain
    """
    base_value = explainer.expected_value
    sv = shap_values[customer_idx]
    feature_vals = X_test.iloc[customer_idx]

    features = X_test.columns.tolist()
    bars = []
    labels = []

    for feat, sv_val, fv in zip(features, sv, feature_vals):
        bars.append(sv_val)
        labels.append(f"{feat} = {fv:.0f}")

    colors = ["#e05c5c" if b > 0 else "#4a90d9" for b in bars]
    y_pos = range(len(bars))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(list(y_pos), bars, color=colors, edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)

    base_prob = _sigmoid(base_value)
    final_prob = _sigmoid(base_value + sum(sv))
    ax.set_title(
        f"SHAP Waterfall — Customer #{X_test.index[customer_idx]}\n"
        f"Base probability: {base_prob:.1%}  →  Final prediction: {final_prob:.1%}",
        fontsize=12
    )
    ax.set_xlabel("SHAP Value (impact on log-odds of churn)", fontsize=10)
    ax.annotate(
        "Red = increases churn risk  |  Blue = decreases churn risk",
        xy=(0.5, -0.15), xycoords="axes fraction",
        ha="center", fontsize=9, color="grey"
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# 7. MAIN — train, evaluate, save plots

if __name__ == "__main__":
    rfm = pd.read_csv("data/rfm_data.csv", index_col="CustomerID")

    # Create label
    rfm = create_churn_label(rfm, recency_threshold=180)

    # Train
    model, X_test, y_test, features = train_churn_model(rfm)

    # Evaluate
    auc, eval_fig = evaluate_model(model, X_test, y_test)
    eval_fig.savefig("data/churn_evaluation.png", dpi=150, bbox_inches="tight")
    print("Saved evaluation plot to data/churn_evaluation.png")

    # Feature importance
    imp_fig = plot_feature_importance(model, features, save_path="data/feature_importance.png")
    print("Saved feature importance plot to data/feature_importance.png")

    plt.show()

    # Example prediction
    print("\n=== Example Prediction ===")
    prob = predict_churn(model, f_score=1, m_score=1)
    print(f"Customer with F=1, M=1 (low frequency, low spend) → Churn probability: {prob:.1%}")

    prob = predict_churn(model, f_score=5, m_score=5)
    print(f"Customer with F=5, M=5 (high frequency, high spend) → Churn probability: {prob:.1%}")
