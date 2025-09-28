import pandas as pd
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# === Part 1 & 2: Load data and simulate salary ===
path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
file_path = f"{path}/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

df["Increment"] = df["PerformanceRating"].apply(lambda x: 1.15 if x == 4 else (1.1 if x == 3 else 1.05))
df["FutureSalary"] = (df["MonthlyIncome"] * df["Increment"]).round(2)

# === Part 3: Train regression models on employees who stayed ===
df_active = df[df["Attrition"] == "No"].copy()

features = [
    "PerformanceRating", "TotalWorkingYears", "YearsAtCompany",
    "JobLevel", "Education", "EnvironmentSatisfaction",
    "JobSatisfaction", "WorkLifeBalance", "MonthlyIncome"
]
X = df_active[features]
y = df_active["FutureSalary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVR": make_pipeline(StandardScaler(), SVR(C=1000, epsilon=100))
}

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return r2, rmse, mape

Reg_Eval = []
for name, model in models.items():
    model.fit(X_train, y_train)
    r2, rmse, mape = evaluate(model, X_test, y_test)
    Reg_Eval.append({
        "Model": name,
        "R2 Score": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape * 100, 2)
    })

Reg_Eval_df = pd.DataFrame(Reg_Eval)
Reg_Eval_file = "C:/Slio/Coding Files/Slio Collage/Sem-4/Ml/Regression_Evaluation.csv"
Reg_Eval_df.to_csv(Reg_Eval_file, index=False)

# === Part 4: Plot Actual vs Predicted Salaries ===
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    axes[i].scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[i].set_title(f'{name} - Actual vs Predicted')
    axes[i].set_xlabel('Actual FutureSalary')
    axes[i].set_ylabel('Predicted FutureSalary')

# Add spacing between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()