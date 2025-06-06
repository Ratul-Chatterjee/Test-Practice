import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.p

df = pd.read_csv('Medicaldataset.csv')
# dataset and labels
X = df.drop(columns=['Result'])
y = df['Result']

le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# K-Fold Cross Validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
cv_accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
cv_precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
cv_recall = cross_val_score(model, X, y, cv=kf, scoring='recall')
cv_f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

# results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print(f"Cross-Validated Accuracy: {cv_accuracy.mean()}")
print(f"Cross-Validated Precision: {cv_precision.mean()}")
print(f"Cross-Validated Recall: {cv_recall.mean()}")
print(f"Cross-Validated F1 Score: {cv_f1.mean()}")

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
    plt.show()

# Root node and entropy calculation
def calculate_entropy(feature_key):
    feature_values = df[feature_key].value_counts(normalize=True)
    entropy = -sum(feature_values * np.log2(feature_values))
    return entropy

root_node = model.tree_.feature[0]
print(f"Root Node: {root_node}")

for column in X.columns:
    print(f"Entropy for {column}: {calculate_entropy(column)}")
