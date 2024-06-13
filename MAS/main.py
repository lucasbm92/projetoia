# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load dataset
df = pd.read_csv('dataset\predictive_maintenance.csv')

# Preprocessing
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Failure Type'] = le.fit_transform(df['Failure Type'])

# Split data into features and target variable
X = df.drop(['UDI', 'Product ID', 'Target'], axis=1)
y = df['Target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
