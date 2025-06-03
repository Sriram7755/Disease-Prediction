import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('dataset.csv')

# Preview data
print("Sample data:\n", df.head())

# Fill missing values with 'None'
df = df.fillna('None')

# Get all unique symptoms
symptom_columns = df.columns[:-1]  # last column is 'Disease'
all_symptoms = set()

for col in symptom_columns:
    all_symptoms.update(df[col].unique())

all_symptoms = sorted([s for s in all_symptoms if s != 'None'])

# Create one-hot encoded features for each symptom
def encode_symptoms(row):
    symptoms = set(row[symptom_columns])
    return [1 if symptom in symptoms else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# Encode the target variable (disease)
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# Save the label encoder for decoding predictions later
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and features
with open("disease_model.pkl", "wb") as f:
    pickle.dump((model, all_symptoms), f)

print("✅ Model trained and saved as 'disease_model.pkl'")
