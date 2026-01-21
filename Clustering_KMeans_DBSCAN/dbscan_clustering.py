from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

def plot_simple(ax, x, y, color, mrkr, lbl):
    ax.scatter(x, y, c=color, marker=mrkr, label=lbl)
    ax.legend(loc='upper left')


X_blobs, Y_blobs = make_blobs(n_samples=800, n_features=2, centers=4, cluster_std=1, shuffle=True, random_state=111)
fbscan_blobs_model = DBSCAN(eps=0.7, min_samples=5, metric='euclidean')
Y_blobs_pred = fbscan_blobs_model.fit_predict(X_blobs)


X_moons, Y_moons =  make_moons(n_samples=400, noise=0.05, random_state=111)
dbscan_moons_model = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')
Y_moons_pred = dbscan_moons_model.fit_predict(X_moons)

X_circles, Y_circles = make_circles(n_samples=800, factor=0.5, noise=0.05, random_state=111)
dbscan_cicrcles_model = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
Y_circles_pred = dbscan_cicrcles_model.fit_predict(X_circles)

fig , axes = plt.subplots(2,3,figsize=(15,5))

#plotting True clusters
#plotting the DBSCAN prediction for blobs dataset
plot_simple(axes[0,0], X_blobs[Y_blobs==0,0], X_blobs[Y_blobs==0,1], 'green', 'o', 'cluster0')
plot_simple(axes[0,0], X_blobs[Y_blobs==1,0], X_blobs[Y_blobs==1,1], 'blue', 's', 'cluster1')
plot_simple(axes[0,0], X_blobs[Y_blobs==2,0], X_blobs[Y_blobs==2,1], 'red', '*', 'cluster2')
plot_simple(axes[0,0], X_blobs[Y_blobs==3,0], X_blobs[Y_blobs==3,1], 'steelblue', 'v', 'cluster3')
#plotting the DBSCAN prediction for moons dataset
plot_simple(axes[0,1], X_moons[Y_moons==0,0], X_moons[Y_moons==0,1], 'green', 'o', 'cluster0')
plot_simple(axes[0,1], X_moons[Y_moons==1,0], X_moons[Y_moons==1,1], 'blue', 's', 'cluster1')
#plotting the DBSCAN prediction for cricles dataset
plot_simple(axes[0,2], X_circles[Y_circles==0,0], X_circles[Y_circles==0,1], 'green', 'o', 'cluster0')
plot_simple(axes[0,2], X_circles[Y_circles==1,0], X_circles[Y_circles==1,1], 'blue', 's', 'cluster1')

#plotting predictions
#plotting the DBSCAN prediction for blobs dataset
plot_simple(axes[1,0], X_blobs[Y_blobs_pred==0,0], X_blobs[Y_blobs_pred==0,1], 'green', 'o', 'cluster0 predicted')
plot_simple(axes[1,0], X_blobs[Y_blobs_pred==1,0], X_blobs[Y_blobs_pred==1,1], 'blue', 's', 'cluster1 predicted')
plot_simple(axes[1,0], X_blobs[Y_blobs_pred==2,0], X_blobs[Y_blobs_pred==2,1], 'red', '*', 'cluster2 predicted')
plot_simple(axes[1,0], X_blobs[Y_blobs_pred==3,0], X_blobs[Y_blobs_pred==3,1], 'steelblue', 'v', 'cluster3 predicted')
plot_simple(axes[1,0], X_blobs[Y_blobs_pred==-1,0], X_blobs[Y_blobs_pred==-1,1], 'grey', '^', 'noise predicted')
#plotting the DBSCAN prediction for moons dataset
plot_simple(axes[1,1], X_moons[Y_moons_pred==0,0], X_moons[Y_moons_pred==0,1], 'green', 'o', 'cluster0 predicted')
plot_simple(axes[1,1], X_moons[Y_moons_pred==1,0], X_moons[Y_moons_pred==1,1], 'blue', 's', 'cluster1 predicted')
plot_simple(axes[1,1], X_moons[Y_moons_pred==-1,0], X_moons[Y_moons_pred==-1,1], 'grey', 'v', 'noise predicted')
#plotting the DBSCAN prediction for cricles dataset
plot_simple(axes[1,2], X_circles[Y_circles_pred==0,0], X_circles[Y_circles_pred==0,1], 'green', 'o', 'cluster0 predicted')
plot_simple(axes[1,2], X_circles[Y_circles_pred==1,0], X_circles[Y_circles_pred==1,1], 'blue', 's', 'cluster1 predicted')
plot_simple(axes[1,2], X_circles[Y_circles_pred==-1,0], X_circles[Y_circles_pred==-1,1], 'grey', 'v', 'noise predicted')


plt.legend()
plt.tight_layout()
plt.show()







