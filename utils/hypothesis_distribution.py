import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
import math

def classify(instance):
    pos = [ x + [0.3, 0.3] for x in instance if np.dot([1,1],x) > 0 ]
    neg = [ x - [0.3, 0.3]for x in instance if np.dot([1,1],x) < 0 ]

    return np.array(pos), np.array(neg)


def labeling(pos_points, neg_points):
    points = np.concatenate((pos_points, neg_points))
    dim = len(pos_points[0])
    
    num_pos =  np.size(pos_points, axis = 0)
    num_neg =  np.size(neg_points, axis = 0)
    
    labels = [[1]  * num_pos + [-1]  * num_neg ]
    labels = np.array(labels).reshape((-1,1))
    
    return points, labels

def distance(pt, h):    
    #print(pt)
    #print(h)
    a = abs( np.dot(pt, h[:2] ) )  
    b = math.sqrt( np.linalg.norm(h[:2]) ) 
    
    return a/b

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

def plot_distribution(plt, h_set):
    x = np.arange(-1, 1.1, 0.1)
    y = np.arange(-1, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.ones((len(x), len(y)))

    #print(h_set)    
    for i in range(len(x)):
        for j in range(len(y)):
            for h in h_set:
                Z[j][i] += distance([j,i],h)
            
    #Z = Z / Z.sum()
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
    pos, neg = classify(instance)

    # for svm
    points, labels = labeling(pos, neg)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(points, labels.ravel() )
    coef = np.concatenate((clf.coef_[0], clf.intercept_)) 
    sample_hypothesis = [ h - coef for h in np.random.randn(1000,3)]
    sel_hypothesis = select_hypothesis(sample_hypothesis, pos, neg) 
    #print(sel_hypothesis)


    # for plot
    plt.xlim(-1,1)
    plt.ylim(-1,1)

    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    plot_hyperplane(plt, [coef])
    plot_hyperplane(plt, sel_hypothesis)
    plot_distribution(plt, sel_hypothesis)
    plt.show()




