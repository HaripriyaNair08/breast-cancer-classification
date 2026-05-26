import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#loading data
columns = ["ID", "Diagnosis", "radius_mean", "texture_mean", "perimeter_mean", 
           "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
           "concave_points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", 
           "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 
           "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", 
           "radius_worst", "texture_worst", "perimeter_worst", "area_worst", 
           "smoothness_worst", "compactness_worst", "concavity_worst", 
           "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]

#read data in csv file
df = pd.read_csv("wdbc.data", header=None, names=columns)

#Preprocessing
df["Diagnosis"] = df["Diagnosis"].map({"B" : 0, "M": 1})#assigning binary value to diagnosis type

x = df.drop(columns=["ID", "Diagnosis"])
y = df["Diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 10)

#Decision tree model
dt = DecisionTreeClassifier(random_state=10)
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)

print("Decision Tree Results: -")
print("Accuracy: ", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

#Decision rules
tree_rules = export_text(dt, feature_names=list(x.columns))
print("Decision Tree Rules: -")
print(tree_rules)

#Random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("Random Forest Results: -")
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))

#Comparison visualizations
models = ["Decision Tree", "Random Forest"]
accuracies = [ accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]

#plotting bar graph
plt.figure()
plt.bar(models, accuracies)
plt.ylim(0.0, 1.0)
plt. ylabel("Accuracy")
plt.title("Accuracy Comparison Between Models")
plt.show()

#SHAP (Explainability Method 1)
explainer = shap.TreeExplainer(rf)

shap_values = explainer(x_test)

#plotting SHAP vizualisation 
shap.summary_plot(shap_values.values[:,:,1], x_test, show=True)
plt.show()

#Feature Importance (Explainability Method 2)
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]

#plotting bar graph
plt.figure()
plt.title("Feature Importance (Random Forest)")
plt.bar(range(x.shape[1]), importance[indices])
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
