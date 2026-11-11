
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

# ---- CHANGE THESE 3 LINES FOR YOUR DATASET ----
CSV = "placement.csv"
TARGET = "placement_status"
POSITIVE = "Placed"
DROP = ["student_id", "salary_package_lpa"]
# -----------------------------------------------

df = pd.read_csv(CSV)
df = df.drop(columns=[c for c in DROP if c in df.columns]).dropna(subset=[TARGET])
y = (df[TARGET] == POSITIVE).astype(int)
raw = df.drop(columns=[TARGET])

# schema is used by app.py to build the input form
schema = {}
for c in raw.columns:
    if not pd.api.types.is_numeric_dtype(raw[c]):
        schema[c] = {"type": "cat", "options": sorted(raw[c].dropna().unique().tolist())}
    else:
        schema[c] = {"type": "num", "min": float(raw[c].min()),
                     "max": float(raw[c].max()), "default": float(raw[c].median())}

X = pd.get_dummies(raw, drop_first=True).astype(float)
X = X.fillna(X.median())

print("Rows:", len(X), "| Placed %:", round(y.mean() * 100, 1))

# ---- quick EDA plots ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
y.map({1: "Placed", 0: "Not placed"}).value_counts().plot.bar(ax=ax[0], title="Class balance")
corr = X.assign(placed=y).corr()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[1].set_title("Correlation heatmap")
plt.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.savefig("eda.png", dpi=150); plt.close()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                      eval_metric="logloss", random_state=42)
except ImportError:
    print("xgboost not installed, skipping (pip install xgboost)")

rows, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
    cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1").mean()
    pipe.fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    rows.append({"Model": name, "CV_F1": cv_f1,
                 "Accuracy": accuracy_score(y_te, p),
                 "Precision": precision_score(y_te, p, zero_division=0),
                 "Recall": recall_score(y_te, p, zero_division=0),
                 "F1": f1_score(y_te, p, zero_division=0)})
    fitted[name] = pipe

res = pd.DataFrame(rows).sort_values("F1", ascending=False).round(3)
print("\n", res.to_string(index=False))
res.to_csv("results.csv", index=False)
res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(figsize=(10, 4), ylim=(0, 1.05))
plt.title("Model comparison"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150); plt.close()

best_name = res.iloc[0]["Model"]
best = fitted[best_name]
print("\nBest model:", best_name)

ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, display_labels=["Not placed", "Placed"])
plt.title(f"Confusion matrix: {best_name}"); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# which features matter most (works for any model; SHAP is an optional upgrade)
pi = permutation_importance(best, X_te, y_te, n_repeats=20, random_state=42, scoring="f1")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(10)
imp.plot.barh(title="Top features (permutation importance)", figsize=(8, 5))
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

joblib.dump({"model": best, "columns": list(X.columns), "schema": schema, "name": best_name},
            "model.joblib")
print("Saved model.joblib, results.csv and plots (eda, model_comparison, confusion_matrix, feature_importance)")
