
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("student_placement_datathon.csv")

print(df.head())
print(df.info())



sns.countplot(x="placement_status", data=df)
plt.title("Placement Distribution")
plt.show()

# CGPA vs Placement
sns.boxplot(x="placement_status", y="cgpa", data=df)
plt.title("CGPA vs Placement")
plt.show()

# Internships vs Placement
sns.countplot(x="internships", hue="placement_status", data=df)
plt.title("Internships vs Placement")
plt.show()

# Backlogs vs Placement
sns.countplot(x="backlogs", hue="placement_status", data=df)
plt.title("Backlogs Impact")
plt.show()

# 4. Feature Engineering

# Create new feature
df["total_experience"] = df["internships"] + df["projects"]

# Encode categorical columns
le = LabelEncoder()
df["coding_skill"] = le.fit_transform(df["coding_skill"])
df["communication_skill"] = le.fit_transform(df["communication_skill"])
df["placement_status"] = le.fit_transform(df["placement_status"])

# 5. Split Data
X = df.drop(["student_id", "placement_status"], axis=1)
y = df["placement_status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 6. Logistic Regression Mode
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# 7. Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 8. Feature Importance
feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

feature_importance.plot(kind="bar", title="Feature Importance")
plt.show()
