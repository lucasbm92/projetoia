# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

# Load dataset
df = pd.read_csv('MAS\dataset\predictive_maintenance.csv')

# Preprocessing
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Failure Type'] = le.fit_transform(df['Failure Type'])

# Split data into features and target variable
X = df.drop(['UDI', 'Product ID', 'Target'], axis=1)
y = df['Target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rebalance classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Calculate class weights
weights = class_weight.compute_sample_weight('balanced', y_train)
class_weights = dict(enumerate(weights))

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights)

# Make predictions and evaluate model
y_pred = (model.predict(X_test) > 0.7).astype("int32")
print(classification_report(y_test, y_pred))
