import pandas as pd
from sklearn.model_selection import train_test_split, KFold , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix, log_loss
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
data = pd.read_csv('Medicaldataset.csv')

zero_not_accepted = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
for column in zero_not_accepted:
    data[column] = data[column].replace(0, np.nan)
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.nan, mean)

# Assuming the dataset has features and a target column named 'target'
X = data[['Age' , 'Troponin']]
y = data['Result']

# Step 2: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_val)

# Step 3: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("Validation Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Step 5: Perform K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_index, val_index in kf.split(X):
    X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[val_index]
    y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[val_index]

    model.fit(X_train_kf, y_train_kf)
    y_pred_kf = model.predict(X_val_kf)

    accuracies.append(accuracy_score(y_val_kf, y_pred_kf))
    precisions.append(precision_score(y_val_kf, y_pred_kf))
    recalls.append(recall_score(y_val_kf, y_pred_kf))
    f1_scores.append(f1_score(y_val_kf, y_pred_kf))

print("Cross-validated Accuracy: {:.2f}%".format(np.mean(accuracies) * 100))
print("Cross-validated Precision: {:.2f}".format(np.mean(precisions)))
print("Cross-validated Recall: {:.2f}".format(np.mean(recalls)))
print("Cross-validated F1-Score: {:.2f}".format(np.mean(f1_scores)))


# Step 6: Plot the decision boundary (only for 2D feature space)
def plot_decision_boundary(X, y, model):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='g')
    plt.title('Decision Boundary')
    plt.show()


# Assuming the dataset has only 2 features for plotting
plot_decision_boundary(X_train, y_train, model)
