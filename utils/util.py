import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


# Measure verifiability on 2 dimension
def measureVerifiability(version_space, particles, num_particles):
        # version_space: num_VS * (dimension + 1), particles: num_particles * (dimension + 1)
        if(version_space.shape[0]>100000):
            iterations = int(version_space.shape[0]/100000)
            answer = np.sum(np.dot(version_space[0:100000,:], particles.T)>0, axis=0) # size: num_VS * num_paticles

            for i in range(1, iterations):
                answer += np.sum(np.dot(version_space[100000*i:100000*(i+1),:], particles.T)>0, axis=0) # size: num_VS * num_paticles
            answer += np.sum(np.dot(version_space[100000*iterations:,:], particles.T)>0, axis=0) # size: num_VS * num_paticles
            verified = (answer==version_space.shape[0]) + (answer==0)
            amount = np.sum(answer==version_space.shape[0]) + np.sum(answer==0)

        else:
            answer = np.sum(np.dot(version_space, particles.T)>0, axis=0) # size: num_VS * num_paticles
            verified = (answer==version_space.shape[0]) + (answer==0)
            amount = np.sum(answer==version_space.shape[0]) + np.sum(answer==0)

        return verified, (amount/float(num_particles))

# Find the hyperplanes
def findIntersection(samples, labels, feasible_point):
       constraints = np.concatenate((samples, np.ones((samples.shape[0], 1)) * feasible_point[0][-1]), axis=1)
       constraints = (-1) * labels * constraints
       #print (constraints, feasible_point[0]) # Examples rule out some H in the hypothesis space

       hs = HalfspaceIntersection(constraints, feasible_point[0][:-1])
       hyperplanes = np.concatenate((hs.intersections / -feasible_point[0][-1], -np.ones((hs.intersections.shape[0], 1))), axis=1)
 
       return (hyperplanes)

def getVersionSpace (samples, labels, hyperplanes):
        # indicate which halfspace is positive
        check_condition = np.dot(hyperplanes, samples.T)
        total_size = check_condition.shape[0] * check_condition.shape[1]
        #print (check_condition.shape[0], check_condition.shape[1])
        if (True):#total_size >= 500000000): # using fewer samples is enough and doesn't lead a memory problem
            check_condition = np.dot(hyperplanes, samples[0:samples.shape[1] + 4, :].T)
            temp = ((check_condition > 0) * check_condition - (check_condition < 0) * check_condition >= 10e-8)
            temp = check_condition * temp
            temp = (temp > 0).astype(int) - (temp < 0).astype(int)
            temp = np.sum(temp * labels[0:samples.shape[1] + 4, :].T, axis=1)
            check_condition = (temp > 0).astype(int) - (temp < 0).astype(int)
        else:
            check_condition = np.sign(np.sum(np.sign(check_condition * (np.abs(check_condition) >= 10e-8)) * labels.T, axis=1))

        boundary = check_condition.reshape(-1, 1) * hyperplanes
        return boundary

def in_hull(points, x):
        n_points = points.shape[0]
        n_dim = x.shape[0]
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b, method='interior-point')
        #print (A, A.shape, lp)
        return lp.success


def classify(points,coeff):
    dim = len(coeff) - 1
    if dim != points.shape[1] :
            print("coeff error: dim is",dim,",and dim of the point is",points.shape[1] )
            return
    
    pos_points = []
    neg_points = []

    for point in points:
            #print('coeff',coeff)
            #print('point',point)
            if np.dot(coeff[:dim],point) > coeff[-1] :
                    pos_points.append(point)
            else:
                    neg_points.append(point)

    pos_points = np.array(pos_points)
    neg_points = np.array(neg_points)

    return pos_points, neg_points
    

def select(points, coeff_set):
    # classify is out of this func
    pos_points = points
    neg_points = points

    for i in range(len(coeff_set)):
        #print(i,"iter select")
        #print('pos_pts',type(pos_points),pos_points[:5])
        #print('coeff_set',coeff_set[i])
        if pos_points.size != 0 :
                pos_points,_ = classify(pos_points, coeff_set[i])
        if neg_points.size != 0 :
                _,neg_points = classify(neg_points, coeff_set[i])

    pos_points = np.array(pos_points)
    neg_points = np.array(neg_points)

    return pos_points, neg_points

def labeling( pos_points, neg_points):
    points = np.concatenate((pos_points, neg_points))
    dim = pos_points.shape[1]
    
    num_pos =  np.size(pos_points, axis = 0)
    num_neg =  np.size(neg_points, axis = 0)

    labels = [[1]  * num_pos + [-1]  * num_neg ]
    labels = np.array(labels).reshape((-1,1))
    
    return points, labels 


def classify_2d(instance):
    pos = [ x + [0.3, 0.3] for x in instance if np.dot([1,1],x) > 0 ]
    neg = [ x - [0.3, 0.3]for x in instance if np.dot([1,1],x) < 0 ]

    return np.array(pos), np.array(neg)


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


