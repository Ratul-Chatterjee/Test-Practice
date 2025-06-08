import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('heart_attack_risk_dataset.csv')

label_encoders = {col: LabelEncoder() for col in data.select_dtypes(include=['object']).columns}
data = data.apply(lambda col: label_encoders[col.name].fit_transform(col) if col.name in label_encoders else col)

X = StandardScaler().fit_transform(data.drop(columns='Heart_Attack_Risk'))
y = data['Heart_Attack_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

accuracy_vals = [model.score(X_train, y_train), model.score(X_test, y_test)]
plt.figure(figsize=(10, 6))
plt.plot(['Training', 'Testing'], accuracy_vals, marker='o')
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy - Health Dataset")
plt.show()
