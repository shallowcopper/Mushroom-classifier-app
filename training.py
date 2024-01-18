# model.py
# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Set the path to the CSV file
file_path = Path("C:/Users/prane/OneDrive/Desktop/DS Project/Group10/Group10/Group_10_Raw_Data.csv")

# Read the clean dataset
df = pd.read_csv(file_path, delimiter=';')

# Display the first few tuples of the dataset
df.head()

# Display the column information
df.info()

# Check for the duplicate tuples, if any
duplicate_rows = df.duplicated()
# Count of duplicate rows
print(f"Number of duplicate rows: {duplicate_rows.sum()}")
# Drop the duplicates
df = df.drop_duplicates()
# Checking the shape of the data after dropping duplicates
print("Shape of DataFrame After Removing Duplicates: ", df.shape)

# Delete four features with atleast 85% of missing values
df = df.loc[:, df.columns != 'stem-root']
df = df.loc[:, df.columns != 'veil-type']
df = df.loc[:, df.columns != 'veil-color']
df = df.loc[:, df.columns != 'spore-print-color']
# Delete two features with atleast 45% of missing values in addition to above
df = df.loc[:, df.columns != 'gill-spacing']
df = df.loc[:, df.columns != 'stem-surface']
# df.info()

# Replace missing values in the dataset using Forward Fill
df.fillna(method='ffill', inplace=True)

# Apply 90% Winsorization Technique to handle the outliers
df['cap-diameter'] = winsorize(df['cap-diameter'],(0.05,0.05))
df['stem-height'] = winsorize(df['stem-height'],(0.05,0.05))
df['stem-width'] = winsorize(df['stem-width'],(0.05,0.05))

# Encode all the categorical columns to numerical data
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder class


label_encoder = LabelEncoder()  # Create an instance for the label encoder
encoded_data = pd.DataFrame()  # Create an empty DataFrame
encoding_mapping = {}  # Dictionary to store encoding mapping for each specified column

for column in df.columns:
    encoded_data[column] = label_encoder.fit_transform(df[column])  # Encode all columns

    # Check if the current column is in the specified columns
    if column in ['class','cap-shape', 'cap-surface', 'cap-color',
                  'does-bruise-or-bleed', 'gill-attachment', 'gill-color', 'stem-color', 'has-ring', 'ring-type', 'habitat',
                  'season']:
        encoding_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

df = encoded_data


# Split the features to predicted (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for performance metric of data
def Perf_Metric(Actual_Output, Prediction_Output):
    Accuracy = round(accuracy_score(Actual_Output, Prediction_Output), 3)
    F1Score = round(f1_score(Actual_Output, Prediction_Output), 3)
    Precision = round(precision_score(Actual_Output, Prediction_Output), 3)
    Recall = round(recall_score(Actual_Output, Prediction_Output), 3)
    return [Accuracy, F1Score, Precision, Recall]

# Fit the model from training data using Random Forest
model = RandomForestClassifier()
rf = model.fit(X_train, y_train)

# Model fit with training data
y_train_predict = rf.predict(X_train)
# Make predictions on the test data
y_pred = rf.predict(X_test)

# Evaluate performance metric for training and testing data
Train_PM = Perf_Metric(y_train, y_train_predict)
Test_PM = Perf_Metric(y_test, y_pred)

RF_Train_PM = Train_PM
RF_Test_PM = Test_PM

# Create a table comparing the performance on training and test data
models = pd.DataFrame({
    'Dataset': ["Train", "Test"],
    'Accuracy Score': [Train_PM[0], Test_PM[0]],
    'F1 Score': [Train_PM[1], Test_PM[1]],
    'Precision': [Train_PM[2], Test_PM[2]],
    'Recall': [Train_PM[3], Test_PM[3]],
})
print(models)



# Save the model
model_filename = 'model.pkl'
joblib.dump(rf, model_filename, compress=9)

import sklearn
print(sklearn.__version__)

