import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


data = pd.read_csv('updated_pollution_dataset.csv')

label_encoder = LabelEncoder()
data['Air Quality Levels'] = label_encoder.fit_transform(data['Air Quality Levels'])

X = data.drop(columns=['Air Quality Levels'])
y = data['Air Quality Levels']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

model_linear = SVC(kernel='linear', random_state=42)
model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)
print(f"Linear SVM - Accuracy: {accuracy_score(y_test, y_pred_linear)}\n")
print(f"Linear SVM - Classification Report:\n{classification_report(y_test, y_pred_linear)}")

model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)

y_pred_rbf = model_rbf.predict(X_test)
print(f"\nRBF SVM - Accuracy: {accuracy_score(y_test, y_pred_rbf)}\n")
print(f"RBF SVM - Classification Report:\n{classification_report(y_test, y_pred_rbf)}")
