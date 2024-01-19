from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# Generate 8 random data points
X = np.array([(3,1),(3,-1),(6,1),(6,-1),(1,0),(0,1),(0,-1),(-1,0)])  # 8 samples, 2 features
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])  # Labels

# Initialize SVM with linear kernel
clf = svm.SVC(kernel='linear')

# Fit the SVM to the data
clf.fit(X, y)

# Plot the data and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create a grid of points to plot the decision boundary
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX, YY = np.meshgrid(xx, yy)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, 
            facecolors='none', edgecolors='k')

plt.title('SVM Decision Boundary and Support Vectors (Linear Kernel)')
plt.show()
