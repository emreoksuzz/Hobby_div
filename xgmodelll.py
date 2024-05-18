import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from joblib import dump, load

# Read the CSV file
df = pd.read_csv("Hobby_Data.csv")


df.rename(columns={
    "Academic_Olympiad": "Olympiad_Participation",
    "Loves_School": "School",
    "Academic_Projects": "Projects",
    "Sports_Medal": "Medals"
}, inplace=True)


df.drop_duplicates(inplace=True)


df['Grasp_pow'] = np.where(df['Grasp_pow'] > 3.5, 'high', 'low')


df = pd.get_dummies(df, columns=["Olympiad_Participation", "Scholarship", "School", "Fav_sub", "Projects", "Grasp_pow",
                                 "Medals", "Act_sprt", "Fant_arts", "Won_arts"])


df = df.drop(columns=["Career_sprt", "Won_arts_Maybe", "Fav_sub_Science"]).reset_index(drop=True)


changeit = {"Academics": 0, "Sports": 1, "Arts": 2}
df["Predicted Hobby"] = df["Predicted Hobby"].map(changeit)


X = df.drop("Predicted Hobby", axis=1)
y = df["Predicted Hobby"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


dump(xgb_model, "XGBoost_model.joblib")
