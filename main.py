import pandas as pd
import kagglehub
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# === Part 1&2: Load data and simulate salary ===
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
file_path = f"{path}/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.15 if x == 4 else (1.1 if x == 3 else 1.05))
df["FutureSalary"] = (df["MonthlyIncome"] * df["Increment"]).round(2)

# === Part 3: Train Ridge regression model on employees who stayed ===
df_active = df[df["Attrition"] == "No"].copy()

features = [
    "PerformanceRating", "TotalWorkingYears", "YearsAtCompany",
    "JobLevel", "Education", "EnvironmentSatisfaction",
    "JobSatisfaction", "WorkLifeBalance", "MonthlyIncome"
]
X = df_active[features]
y = df_active["FutureSalary"]

# Train Ridge on entire data (no test split)

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVR": make_pipeline(StandardScaler(), SVR(C=1000, epsilon=100))
}


for name, model in models.items():
    model.fit(X, y)

# === Part 4: Predict Attrition, Filter Likely to Stay, Predict Future Salary ===
df["AttritionEncoded"] = LabelEncoder().fit_transform(df["Attrition"])  # Yes=1, No=0

if df["OverTime"].dtype == object:
    df["OverTime"] = LabelEncoder().fit_transform(df["OverTime"])

attrition_features = [
    "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "TotalWorkingYears",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole"
]
X_attr = df[attrition_features]
y_attr = df["AttritionEncoded"]

# Scale features
scaler = StandardScaler()
X_attr_scaled = scaler.fit_transform(X_attr)

# Logistic Regression trained on full dataset
clf = LogisticRegression()
clf.fit(X_attr_scaled, y_attr)

# Predictions with custom threshold
threshold = 0.32  # gave best roc_auc on test set
p_leave_prob = clf.predict_proba(X_attr_scaled)[:, 1]
custom_pred = (p_leave_prob >= threshold).astype(int)

# P_stay calculated using probabilities
df["P_stay"] = (1 - p_leave_prob).round(2)
df_likely_to_stay = df[custom_pred == 0].copy()  # 0 = predicted to stay

# Predict future salary using Ridge model
X_salary = df_likely_to_stay[features]
future_salary_pred = models["Ridge"].predict(X_salary)
df_likely_to_stay["PredictedFutureSalary"] = future_salary_pred.round(2)

Reg_Sal_Pre_file = "C:/Slio/Coding Files/Slio Collage/Sem-4/Ml/Likely_to_Stay_Salary_Predictions.csv"
df_likely_to_stay[["EmployeeNumber", "Attrition", "MonthlyIncome", "P_stay", "PredictedFutureSalary"]].to_csv(Reg_Sal_Pre_file, index=False)

# === Part 5: Expected Salary Loss (Simple Version) ===
df_likely_to_stay["ExpectedLoss"] = ((1 - df_likely_to_stay["P_stay"]) * df_likely_to_stay["FutureSalary"]).round(2)

output_expected_loss_likely_to_stay = "C:/Slio/Coding Files/Slio Collage/Sem-4/Ml/Expected_Salary_Loss_Likely_to_Stay.csv"
df_likely_to_stay[["EmployeeNumber", "FutureSalary", "P_stay", "ExpectedLoss"]].to_csv(output_expected_loss_likely_to_stay, index=False)

total_loss_likely_to_stay = df_likely_to_stay["ExpectedLoss"].sum().round(2)
print(f"Total Expected Salary Loss (Likely to Stay): ₹{total_loss_likely_to_stay}")