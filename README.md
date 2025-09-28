# HR Attrition and Salary Forecasting

## 📍 Overview
This project leverages the **IBM HR Analytics Attrition Dataset** to perform a two-stage predictive analysis for workforce management:

1. **Attrition Prediction (Classification)**  
   Predict whether an employee is likely to leave the organization using models such as:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest

2. **Salary Forecasting (Regression)**  
   Estimate future salaries of employees predicted to stay, using simulated increments and regression models such as:
   - Ridge Regression  
   - Additional regression models evaluated in comparison  

3. **Expected Salary Loss (Novel Metric)**  
   Introduced a new metric that combines:
   - Attrition probability  
   - Future salary estimates  
   
   ➝ This quantifies the **potential financial risk** to the organization if valuable employees leave.  

---

### 🪛Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

### 📚 Dataset
**Source:** [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
This is available on Kaggle.

---
### 📑 Project Report
A detailed report of the methodology, experiments, and results is available here:  
   [View Report (PDF)]([report/HR_Attrition_Report.pdf](https://github.com/manognachalasani/HR-Attrition-and-Salary-Forecasting/blob/main/report.pdf))

The report includes:  
- Data exploration and preprocessing steps  
- Model architectures and evaluation metrics  
- Comparative performance of classifiers (Logistic Regression, SVM, Random Forest)  
- Regression model analysis for salary forecasting  
- Insights from the **Expected Salary Loss** metric  
