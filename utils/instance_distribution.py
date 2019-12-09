import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

if __name__ == "__main__":
    # generate instance 
    instance = np.random.rand(10,2) - [0.5, 0.5]
    pos, neg = classify(instance)

    # for plot
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    #plt.plot(instance[:,0], instance[:,1], "o")
    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    plot_distribution(plt, pos, neg)
    plt.show()




