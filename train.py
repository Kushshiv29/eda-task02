# train.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv(r"C:\Users\shivaay\Downloads\tested1.csv")

# =============================
# 1. DATA CLEANING
# =============================

# Check for missing values
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Fill missing Age and Fare with median
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Fill missing Cabin with 'Unknown'
df["Cabin"].fillna("Unknown", inplace=True)

# Convert categorical variables
df["Sex"] = df["Sex"].astype("category")
df["Pclass"] = df["Pclass"].astype("category")
df["Embarked"] = df["Embarked"].astype("category")

print("\nMissing values after cleaning:\n", df.isnull().sum())

# =============================
# 2. EXPLORATORY DATA ANALYSIS
# =============================

# Plot 1: Survival count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xticks([0, 1], ["Not Survived", "Survived"])
plt.show()

# Plot 2: Survival by Gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Plot 3: Survival by Pclass
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Plot 4: Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Age"], kde=True, bins=30)
plt.title("Age Distribution")
plt.show()

# Plot 5: Fare Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Fare"], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()

# Plot 6: Correlation Heatmap
numeric_df = df[["Survived", "Age", "SibSp", "Parch", "Fare"]]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
