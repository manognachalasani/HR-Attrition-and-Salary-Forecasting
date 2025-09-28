# logistic regression + custom threshold + no smote (smote decreases 8% f1 & auc decreases )
import random
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    precision_recall_curve,
    auc as auc_pr
)

# Load dataset
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
file_path = f"{path}/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions with custom threshold
threshold = 0.32  # Modify this value to change the threshold
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_prob >= threshold).astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_prob)

print(f"=== Logistic Regression Evaluation {threshold:.2f} ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()

# Calculate Precision-Recall Curve and AUC
precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
auc_pr_score = auc_pr(recall_pr, precision_pr)
print(f"\nPrecision-Recall AUC Score: {auc_pr_score:.4f}")

# Plot Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall_pr, precision_pr, label=f'Logistic Regression (PR AUC = {auc_pr_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.show()

# Random sample prediction
sample_index = random.randint(0, len(X_test_scaled) - 1)
sample = X_test_scaled[[sample_index]]
sample_prediction = model.predict(sample)[0]
print(f"\nSample Person Prediction (0 = Stay, 1 = Leave): {sample_prediction}")