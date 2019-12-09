import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC

def classify(instance):
    pos = [ x + [0.3, 0.3] for x in instance if np.dot([1,1],x) > 0 ]
    neg = [ x - [0.3, 0.3]for x in instance if np.dot([1,1],x) < 0 ]

    return np.array(pos), np.array(neg)

def plot_distribution(plt, pos, neg):
    x = np.arange(-1, 1.1, 0.1)
    y = np.arange(-1, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.ones((len(x), len(y)))

    pos_kde = stats.kde.gaussian_kde(pos.T)
    neg_kde = stats.kde.gaussian_kde(neg.T)

    for i in range(len(x)):
        for j in range(len(y)):
            pos_prob = pos_kde(np.array([x[i], y[j] ]).T)
            neg_prob = neg_kde(np.array([x[i], y[j] ]).T)
            Z[j][i] = pos_prob / (pos_prob + neg_prob) 
            
    #Z = Z / Z.sum()


    plt.contourf(X, Y, Z, 100, alpha=.5, cmap=plt.get_cmap('jet'))

def labeling(pos_points, neg_points):
    points = np.concatenate((pos_points, neg_points))
    dim = len(pos_points[0])
    
    num_pos =  np.size(pos_points, axis = 0)
    num_neg =  np.size(neg_points, axis = 0)
    
    labels = [[1]  * num_pos + [-1]  * num_neg ]
    labels = np.array(labels).reshape((-1,1))
    
    return points, labels

def plot_hyperplane(plt, coef, color = 'k'):
    # for hyperplane ax+by+cz=d
    a,b,c = coef[0],coef[1],coef[2]
    
    x = np.linspace(-1,1,2)
    y = (c - a*x ) / b
    plt.plot(x,y)
    

if __name__ == "__main__":
    # generate instance 
    instance = np.random.rand(10,2) - [0.5, 0.5]
    pos, neg = classify(instance)

    # for svm
    points, labels = labeling(pos, neg)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(points, labels.ravel() )
    coef = np.concatenate((clf.coef_[0], clf.intercept_)) 


    # for plot
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    #plt.plot(instance[:,0], instance[:,1], "o")
    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    plot_hyperplane(plt, coef)
    #plot_distribution(plt, pos, neg)
    plt.show()




