import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump,load

pd.set_option('display.max_columns', None)
df=pd.read_csv("credit_data.csv")


import pandas as pd
import numpy as np
import seaborn as sns

# Adjust the file path to include the file extension
df = pd.read_csv("Hobby_Data.csv")


df.rename(columns={
    "Academic_Olympiad": "Olympiad_Participation",
    "Loves_School": "School",
    "Academic_Projects": "Projects",
    "Sports_Medal": "Medals"
}, inplace=True)


df.drop_duplicates(inplace=True)




df['Grasp_pow']=df['Grasp_pow'].apply(lambda x:"high" if x > 3.5 else "low")

df=pd.get_dummies(df,columns=["Olympiad_Participation","Scholarship",'School','Fav_sub','Projects', 'Grasp_pow','Medals',
                              'Act_sprt', 'Fant_arts', 'Won_arts'])

df=df.drop(columns=["Career_sprt","Won_arts_Maybe","Fav_sub_Science"]).reset_index(drop=True)



changeit={"Academics":0,"Sports":1,"Arts":2}

df["Predicted Hobby"] = df["Predicted Hobby"].replace(changeit)
print(df["Predicted Hobby"].value_counts())



"""
Index(['Olympiad_Participation', 'Scholarship', 'School', 'Fav_sub',
       'Projects', 'Grasp_pow', 'Time_sprt', 'Medals', 'Career_sprt',
       'Act_sprt', 'Fant_arts', 'Won_arts', 'Time_art', 'Predicted Hobby'],
      dtype='object')"""



X=df.drop("Predicted Hobby",axis=1)
y=df["Predicted Hobby"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

svm_model = SVC(kernel='linear', probability=True)  # Using a linear kernel for interpretability
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

feature_importances = np.abs(svm_model.coef_[0])
feature_names = df.drop("Predicted Hobby", axis=1).columns

importances_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
importances_df = importances_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importances_df)
plt.title("Feature Importances in SVM")
plt.show()


dump(svm_model,"SVM_new_model")

