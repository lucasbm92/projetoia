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
import matplotlib.pyplot as plt

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
history = model.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()

# Make predictions and evaluate model
y_pred = (model.predict(X_test) > 0.7).astype("int32")
print(classification_report(y_test, y_pred))
