import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy import stats
from util import labeling, classify_2d, select_hypothesis
from instance_distribution import plot_distribution as id
from hypothesis_distribution import plot_distribution as hd
from hypothesis_distribution import plot_hyperplane as hd_plane
from verifiability_distribution import plot_distribution as vd
from verifiability_expectation import plot_expectation as ve

if __name__ == "__main__":
    # generate instance 
    instance = np.random.rand(10,2) - [0.5, 0.5]
    pos, neg = classify_2d(instance)

    # for svm
    points, labels = labeling(pos, neg)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(points, labels.ravel() )
    coef = np.concatenate((clf.coef_[0], clf.intercept_)) 
    sample_hypothesis = [ h - coef for h in np.random.randn(20,3)]
    sel_hypothesis = select_hypothesis(sample_hypothesis, pos, neg) 

    # for plot
    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    plt.gca().set_title("instance distribution")
    id(plt, pos, neg)
    
    plt.subplot(2,2,2)
    plt.gca().set_title("hypothesis_distribution")
    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    hd_plane(plt, sel_hypothesis[:10])
    hd(plt, sel_hypothesis)
    
    plt.subplot(2,2,3)
    plt.gca().set_title("verifiability distribution")
    plt.plot(pos[:,0], pos[:,1], "or")
    plt.plot(neg[:,0], neg[:,1], "ob")
    vd(plt, sel_hypothesis)
    
    plt.subplot(2,2,4)
    plt.gca().set_title("verifiability expectation")
    ve(plt, instance)

    import sys
    iter = sys.argv[1]
    plt.savefig("../result/all_distribution_" + iter + ".png")
    #plt.show()


