import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data               # Features (4)
y = iris.target             # Target classes (0,1,2)

# Reduce to 2 features for visualization later
X_vis = X[:, :2]

# Normalize both full and 2D features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_vis_scaled = scaler.fit_transform(X_vis)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_vis_train, X_vis_test, _, _ = train_test_split(X_vis_scaled, y, test_size=0.2, random_state=42)

# Try different values of K and print accuracy
print("🔍 KNN Accuracy for K=1 to K=10:")
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k} → Accuracy: {acc:.2f}")

# Choose the best K manually (e.g., 3 here)
best_k = 3
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred_final = knn_final.predict(X_test)

# Accuracy and Confusion Matrix
print(f"\n✅ Final Accuracy with K={best_k}: {accuracy_score(y_test, y_pred_final):.2f}")
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (K={best_k})')
plt.show()

# Function to visualize decision boundaries using first 2 features
def plot_decision_boundaries(X, y, model, title):
    h = 0.02  # step size in mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['red', 'green', 'blue']

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot data points
    for i, color in zip(range(3), cmap_bold):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    c=color, label=iris.target_names[i],
                    edgecolor='k', s=40)

    plt.title(title)
    plt.xlabel('Feature 1 (normalized)')
    plt.ylabel('Feature 2 (normalized)')
    plt.legend()
    plt.show()

# Train model for visualization using only 2 features
knn_vis = KNeighborsClassifier(n_neighbors=best_k)
knn_vis.fit(X_vis_train, y_train)

# Visualize decision boundary
plot_decision_boundaries(X_vis_scaled, y, knn_vis, f"Decision Boundaries (K={best_k})")
