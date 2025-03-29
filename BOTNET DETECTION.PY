import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate a synthetic dataset similar to botnet attack data
X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
dataset = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
dataset['Target'] = y

# Show the first 10 rows of the dataset
print("Dataset Sample:")
print(dataset.head(10))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Visualization
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.histplot(dataset.iloc[:, 0], bins=30, kde=True, color='blue')
plt.title("Histogram of Feature 1")

plt.subplot(2, 3, 2)
plt.plot(dataset.index[:100], dataset.iloc[:100, 1], color='red')
plt.title("Line Chart of Feature 2")

plt.subplot(2, 3, 3)
dataset['Target'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title("Bar Chart of Class Distribution")
plt.xticks(ticks=[0,1], labels=['Normal', 'Botnet'], rotation=0)

plt.subplot(2, 3, 4)
sns.boxplot(x=dataset['Feature_3'], color='green')
plt.title("Box Plot of Feature 3")

plt.subplot(2, 3, 5)
sns.scatterplot(x=dataset['Feature_4'], y=dataset['Feature_5'], hue=dataset['Target'])
plt.title("Scatter Plot of Feature 4 vs Feature 5")

plt.subplot(2, 3, 6)
dataset['Target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['blue', 'orange'])
plt.title("Pie Chart of Class Proportion")

plt.tight_layout()
plt.show()

# Heatmap of Feature Correlations
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap of Feature Correlations")
plt.show()

# Train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
ocsvm.fit(X_train_scaled)
y_pred_ocsvm = ocsvm.predict(X_test_scaled)
y_pred_ocsvm = np.where(y_pred_ocsvm == 1, 0, 1)

# Autoencoder
autoencoder = Sequential([
    Dense(20, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(10, activation='relu'),
    Dense(20, activation='relu'),
    Dense(X_train_scaled.shape[1], activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32, verbose=0)

encoder = Sequential(autoencoder.layers[:2])
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

rf_encoded_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_encoded_model.fit(X_train_encoded, y_train)
y_pred_encoded = rf_encoded_model.predict(X_test_encoded)

# Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate Models
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_ocsvm, "One-Class SVM")
evaluate_model(y_test, y_pred_encoded, "Autoencoder + Random Forest")