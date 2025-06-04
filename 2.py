import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data
dataset = pd.read_csv('Medicaldataset.csv')

# Handle columns with zeros and replace them with NaN, then fill with mean
zero_not_accepted = ['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.nan)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.nan, mean)

# Define X (features) and y (target)
X = dataset[['Age', 'Troponin']]
y = dataset['Result']

# If the labels are 0 and 1, convert them to 'negative' and 'positive'
y = y.replace({0: 'negative', 1: 'positive'})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_precision = precision_score(y_train, y_pred_train, pos_label='positive')
test_precision = precision_score(y_test, y_pred_test, pos_label='positive')
train_recall = recall_score(y_train, y_pred_train, pos_label='positive')
test_recall = recall_score(y_test, y_pred_test, pos_label='positive')
train_f1 = f1_score(y_train, y_pred_train, pos_label='positive')
test_f1 = f1_score(y_test, y_pred_test, pos_label='positive')

train_loss = log_loss(y_train, pipeline.predict_proba(X_train))
test_loss = log_loss(y_test, pipeline.predict_proba(X_test))

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Training Loss:", train_loss)
print("Test Loss:", test_loss)
print("Training Precision:", train_precision)
print("Test Precision:", test_precision)
print("Training Recall:", train_recall)
print("Test Recall:", test_recall)
print("Training F1-Score:", train_f1)
print("Test F1-Score:", test_f1)

# Decision boundary plot (only if X_train has 2 features)
if X_train.shape[1] == 2:
    # Convert X_train to numpy array if it is still a DataFrame
    X_train_array = np.array(X_train)

    x_min, x_max = X_train_array[:, 0].min() - 1, X_train_array[:, 0].max() + 1
    y_min, y_max = X_train_array[:, 1].min() - 1, X_train_array[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Make predictions over the mesh grid and reshape for contour plot
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Ensure Z contains numeric values for contour plotting
    Z_numeric = np.where(Z == 'positive', 1, 0)  # Convert 'positive' -> 1 and 'negative' -> 0

    # Plot the decision boundary
    plt.contourf(xx, yy, Z_numeric, alpha=0.8)
    plt.scatter(X_train_array[:, 0], X_train_array[:, 1], c=y_train.map({'negative': 0, 'positive': 1}), edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.show()

# Cross-validation - using pipeline to automatically handle scaling and fitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
cv_precision = cross_val_score(pipeline, X, y, cv=kf, scoring='precision')
cv_recall = cross_val_score(pipeline, X, y, cv=kf, scoring='recall')
cv_f1 = cross_val_score(pipeline, X, y, cv=kf, scoring='f1')

print("Cross-Validation Accuracy:", np.mean(cv_accuracy))
print("Cross-Validation Precision:", np.mean(cv_precision))
print("Cross-Validation Recall:", np.mean(cv_recall))
print("Cross-Validation F1-Score:", np.mean(cv_f1))
