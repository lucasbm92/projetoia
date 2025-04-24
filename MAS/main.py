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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.optimizers import Adam

# Carrega o dataset
df = pd.read_csv('MAS\dataset\predictive_maintenance.csv')

# Pré-processamento
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Failure Type'] = le.fit_transform(df['Failure Type'])

# Divide o dataset em features e target
X = df.drop(['UDI', 'Product ID', 'Target'], axis=1)
y = df['Target']

# Divide o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Realiza o balanceamento das classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Calcula os pesos das classes
weights = class_weight.compute_sample_weight('balanced', y_train)
class_weights = dict(enumerate(weights))

# # Define a arquitetura da rede neural
# model = Sequential()
# model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# learning_rate = 0.01 # Define a taxa de aprendizagem

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
learning_rate = 0.02

# Cria o otimizador Adam com a taxa de aprendizagem definida
opt = Adam(learning_rate=learning_rate)

# Compila o modelo com o otimizador personalizado
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Treina o modelo com pesos das classes e armaena o histórico
history = model.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights)

# Plota os valores de treino e accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Plota os valores de treino e loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Ajusta o layout e exibe os gráficos
plt.tight_layout()
plt.show(block=False)

# Faz a predição e exibe o relatório de classificação
y_pred = (model.predict(X_test) > 0.7).astype("int32")
print(classification_report(y_test, y_pred))

# Computa a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()