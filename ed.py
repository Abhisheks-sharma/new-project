 # Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Loading the dataset
df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv", header=1) 
print(df.head())  # Display first few rows
print(df.describe())  # Summary statistics
print(df.info())  # Dataset info

# Checking for missing values
print(df[df.isnull().any(axis=1)])

# Assigning region labels (0 for the first region, 1 for the second)
df.loc[:122, "region"] = 0
df.loc[122:, "region"] = 1

# Casting region column to integer
df["region"] = df["region"].astype(int)
data = df

# Checking for missing values again
print(data.isnull().sum())

# Dropping rows with missing values and resetting index
data = data.dropna().reset_index(drop=True)
print(data.isnull().sum())

# Dropping a specific row by index (if necessary)
data = data.drop(122).reset_index(drop=True)

# Cleaning up column names (removing leading/trailing spaces)
data.columns = data.columns.str.strip()
print(data.columns)

# Converting specific columns to integer types
data[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = data[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

# Identify object type columns
objects = [feature for feature in data.columns if data[feature].dtypes == "object"]
print(objects)

# Converting object type columns (except "Classes") to float
for col in objects:
    if col != "Classes":
        data[col] = data[col].astype(float)

# Display the cleaned data
print(data.head())

# Saving the cleaned dataset to CSV
data.to_csv("Algerian_forest_fires_dataset.csv", index=False)

# Feature Engineering: Dropping unnecessary columns
df_copy = data.drop(["year", "day", "month"], axis=1)

# Converting "Classes" column to binary (0 for 'not fire', 1 for 'fire')
df_copy["Classes"] = np.where(df_copy["Classes"].str.contains("not fire"), 0, 1)
print(df_copy["Classes"].value_counts())

# Defining features (X) and target (y)
X = df_copy.drop(["FWI"], axis=1)  # Dropping the FWI column (target)
y = df_copy["FWI"]  # Target variable

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Selection: Dropping highly correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set to hold correlated columns
    corr_matrix = dataset.corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:  # Threshold for correlation
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    
    return col_corr

# Identifying highly correlated features (correlation > 0.85)
corr_features = correlation(X_train, 0.85)
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training: Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
y_pred_lr = linreg.predict(X_test_scaled)

# Model Evaluation: Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - R2: {r2_lr}, MAE: {mae_lr}")

# Model Training: Ridge Regression
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Model Evaluation: Ridge Regression
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - R2: {r2_ridge}, MAE: {mae_ridge}")

# Saving the trained models and scaler to disk
import pickle
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(ridge, open("ridge.pkl", "wb"))
