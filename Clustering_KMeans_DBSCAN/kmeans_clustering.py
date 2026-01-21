from sklearn.cluster import KMeans
from sklearn.datasets import make_moons, make_blobs, make_circles
import numpy as np
import matplotlib.pyplot as plt

    
def plot_simple(ax, x, y, color, mrkr, lbl):
    ax.scatter(x, y, c=color, marker=mrkr, label= lbl)
    ax.legend(loc='upper left')
        
X_moons, Y_moons = make_moons(n_samples=600, noise=0.01, random_state=0)
X_blobs, Y_blobs = make_blobs(n_samples=600, n_features=2, centers=4, cluster_std=1, shuffle=True, random_state=0)
X_circles, Y_circles = make_circles(n_samples=600, noise=0.01, random_state=0)


km_model_moons = KMeans(n_clusters=2, init='random', n_init=20, max_iter=200, tol=1e-05, random_state=111)
Y_moons_pred = km_model_moons.fit_predict(X_moons)

km_model_blobs = KMeans(n_clusters=4, init='random', n_init=20, max_iter=200, tol=1e-05, random_state=111)
Y_blobs_pred = km_model_blobs.fit_predict(X_blobs)

km_model_circles = KMeans(n_clusters=2, init='random', n_init=20, max_iter=200, tol=1e-05, random_state=111)
Y_circles_pred = km_model_circles.fit_predict(X_circles)



fig, axes = plt.subplots(2,3,figsize=(15,8))

#plot true clusters
plot_simple(axes[0,0], X_moons[Y_moons == 0,0], X_moons[Y_moons == 0,1], 'blue', 'o', 'cluster0')
plot_simple(axes[0,0], X_moons[Y_moons == 1,0], X_moons[Y_moons == 1,1], 'green', 's', 'cluster1')

plot_simple(axes[0,1], X_blobs[Y_blobs == 0,0],X_blobs[Y_blobs == 0, 1], 'blue', 'o', 'cluster0')
plot_simple(axes[0,1], X_blobs[Y_blobs == 1,0],X_blobs[Y_blobs == 1, 1], 'green', 's', 'cluster1')
plot_simple(axes[0,1], X_blobs[Y_blobs == 2,0],X_blobs[Y_blobs == 2, 1], 'red', 'v', 'cluster2')
plot_simple(axes[0,1], X_blobs[Y_blobs == 3,0],X_blobs[Y_blobs == 3, 1], 'steelblue', '*', 'cluster3')

plot_simple(axes[0,2], X_circles[Y_circles == 0,0], X_circles[Y_circles == 0,1], 'blue', 'o', 'cluster0')
plot_simple(axes[0,2], X_circles[Y_circles == 1,0], X_circles[Y_circles == 1,1], 'green', 's', 'cluster1')

#plot predicted clusters
plot_simple(axes[1,0], X_moons[Y_moons_pred == 0,0], X_moons[Y_moons_pred == 0,1], 'blue', 'o', 'cluster0 predicted')
plot_simple(axes[1,0], X_moons[Y_moons_pred == 1,0], X_moons[Y_moons_pred == 1,1], 'green', 's', 'cluster1 predicted')

plot_simple(axes[1,1], X_blobs[Y_blobs_pred == 0,0],X_blobs[Y_blobs_pred == 0, 1], 'blue', 'o', 'cluster0 predicted')
plot_simple(axes[1,1], X_blobs[Y_blobs_pred == 1,0],X_blobs[Y_blobs_pred == 1, 1], 'green', 's', 'cluster1 predicted')
plot_simple(axes[1,1], X_blobs[Y_blobs_pred == 2,0],X_blobs[Y_blobs_pred == 2, 1], 'red', 'v', 'cluster2 predicted')
plot_simple(axes[1,1], X_blobs[Y_blobs_pred == 3,0],X_blobs[Y_blobs_pred == 3, 1], 'steelblue', '*', 'cluster3 predicted')

plot_simple(axes[1,2], X_circles[Y_circles_pred == 0,0], X_circles[Y_circles_pred == 0,1], 'blue', 'o', 'cluster0 predicted')
plot_simple(axes[1,2], X_circles[Y_circles_pred == 1,0], X_circles[Y_circles_pred == 1,1], 'green', 's', 'cluster1 predicted')

plt.tight_layout()
plt.show()
