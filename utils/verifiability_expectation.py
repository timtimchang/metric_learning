import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
import math

def classify(instance):
    pos = [ x + [0.3, 0.3] for x in instance if np.dot([1,1],x) > 0 ]
    neg = [ x - [0.3, 0.3]for x in instance if np.dot([1,1],x) < 0 ]

    return np.array(pos), np.array(neg)

def verify(pt, h_set):
    num_h = len(h_set)
    count = 0.0
    
    for h in h_set :
        #print(h)
        if np.dot(h[:2], pt) > h[2]:
            count +=1

    return count / num_h

def labeling(pos_points, neg_points):
    points = np.concatenate((pos_points, neg_points))
    dim = len(pos_points[0])
    
    num_pos =  np.size(pos_points, axis = 0)
    num_neg =  np.size(neg_points, axis = 0)
    
    labels = [[1]  * num_pos + [-1]  * num_neg ]
    labels = np.array(labels).reshape((-1,1))
    
    return points, labels

def select_hypothesis(h_set, pos_set, neg_set):
    h_set = [h  if h[0] > 0 else -h for h in h_set]
    sel_h_set = []

    for h in h_set:
        choose = True
        for pos in pos_set:    
            if np.dot(h[:2],pos) < h[2] :
                choose = False
                break

        if choose is False : continue

        for neg in neg_set:
            if  np.dot(h[:2],neg) > h[2] : 
                choose = False
                break
        
        if choose is True : sel_h_set.append(h)

    return sel_h_set    

def plot_expectation(plt, instance):
    # for classify
    pos, neg = classify(instance)

    # for svm
    points, labels = labeling(pos, neg)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(points, labels.ravel() )
    coef = np.concatenate((clf.coef_[0], clf.intercept_)) 

    #for sv
    sv = clf.support_vectors_

    # select hypothesis
    sample_hypothesis = [ h - coef for h in np.random.randn(100,3)]
    sel_hypothesis = select_hypothesis(sample_hypothesis, pos, neg) 


    # for distribution
    #pos_sv , neg_sv = classify(sv)
    #print(pos)
    #print(pos_sv)
    pos_kde = stats.kde.gaussian_kde(pos.T)
    neg_kde = stats.kde.gaussian_kde(neg.T)

    
    # for plotting
    x = np.arange(-1, 1.1, 0.1)
    y = np.arange(-1, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.ones((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            pos_prob = pos_kde(np.array([x[i], y[j] ]).T)
            pos_ver_prob = pos_prob * ( 1 - verify([x[i], y[j]],sel_hypothesis) )

            neg_prob = neg_kde(np.array([x[i], y[j] ]).T)
            neg_ver_prob = neg_prob * verify([x[i], y[j]],sel_hypothesis) 

            Z[j][i] = max(pos_ver_prob, neg_ver_prob)
            
    #Z = Z / Z.sum()
    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    plt.contourf(X, Y, Z, 100, alpha=.5, cmap=plt.get_cmap('jet'))

def plot_hyperplane(plt, coef_set, color = 'k'):
    for coef in coef_set:
        # for hyperplane ax+by+cz=d
        a,b,c = coef[0],coef[1],coef[2]
        
        x = np.linspace(-1,1,2)
        y = (c - a*x ) / b
        plt.plot(x,y)


if __name__ == "__main__":
    # generate instance 
    instance = np.random.rand(10,2) - [0.5, 0.5]

    # for plot
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    #plot_hyperplane(plt, [coef])
    #plot_hyperplane(plt, sel_hypothesis)
    plot_expectation(plt, instance)


    import sys
    iter = sys.argv[1]
    plt.savefig("../result/verifiability_expectation_" + iter + ".png")
    #plt.show()



