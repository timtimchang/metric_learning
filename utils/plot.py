import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


def plot_points(ax,points,color):
#	print("convex 3d points",points)
	ax.plot(points[:,0], points[:,1],points[:,2], color + "o")

def plot_surface(ax,points,color = (0,0,0,0),alpha = 0.3):
	# plot the surface
 	# simplices : Indices of points forming the simplical facets of the convex hull.
	hull = ConvexHull(points)
	for s in hull.simplices:
		ax.plot_trisurf(points[s, 0], points[s, 1], points[s, 2],
						 alpha = alpha, color=color, linewidth=0.2, antialiased=True)
		
def plot_point_and_corner(ax,points,hull,color):
	# black for points and color for corners
	ax.plot(points[:,0], points[:,1],points[:,2], "ko")

	ax.plot(points[hull.vertices[:],0],
			points[hull.vertices[:],1],
			points[hull.vertices[:],2],
			color + "o")

def plot_convex(ax,points,color):
	# for convex hull	
	hull = ConvexHull(points)
	plot_point_and_corner(ax,points,hull,color)
	plot_surface(ax,points) 

def plot_augment_convex(ax,points,new_point,color):
	# for convex hull	
	hull = ConvexHull(points)
	plot_point_and_corner(ax,points,hull,color)
	plot_surface(ax,points,hull) 

	# for new convex hull
	new_hull = ConvexHull( new_point )
	plot_surface(ax,new_point,new_hull,color=(1,1,0,0),alpha = 0.3)

	# plot diminish points 
	diminish_points = find_diminish_points(points,new_point) 
	if len(diminish_points) > 0 :
		ax.plot(diminish_points[:,0],
				diminish_points[:,1],
				diminish_points[:,2],
				"kx")

	# plot new point
	ax.plot(new_point[:,0],new_point[:,1],new_point[:,2],color + "o")


def divide_labels(samples, labels):
	positive_index = np.where(labels[:, 0] == True)
	negitive_index = np.where(labels[:, 0] == False)
	
	positive_set_t = np.take(samples, positive_index, axis=0)
	negitive_set_t = np.take(samples, negitive_index, axis=0)
	
	positive_set = positive_set_t[0]
	negitive_set = negitive_set_t[0]
	# print("divide_labels positive_set",positive_set)
	
	return positive_set, negitive_set

def find_diminish_points(points,new_point):
	# for convex hull	
	hull = ConvexHull(points)

	# for new convex hull 
	new_points = np.concatenate((new_point,points),axis = 0)
	new_hull = ConvexHull( new_points )

	# diminish point
	diminish_points = np.array([ i for i in new_points[new_hull.vertices] if i not in points[hull.vertices] ])  
	#print("points:",points)
	#print("new points:",new_points)
	#print("diminish_points:",diminish_points)

	return diminish_points

def plot_hyperplane(ax, coeff, color = 'k'):
    # for hyperplane ax+by+cz=d
    a,b,c,d = coeff[0],coeff[1],coeff[2],0

    x = np.linspace(-1,1,2)
    y = np.linspace(-1,1,2)

    X,Y = np.meshgrid(x,y)
    Z = (d - a*X - b*Y) / c

    surf = ax.plot_surface(X, Y, Z,color = color, alpha = 0.5)	

def plot_sphere(ax,color ='k'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha = 0.5)
    
def plot_sphere_map(ax,points):
    mesh = 200

    X = np.arange(-1, 1, 2/mesh)
    Y = np.arange(-1, 1, 2/mesh)
    Z = np.zeros((len(X), len(Y)))

    u = np.linspace(0, 2 * np.pi, mesh)
    v = np.linspace(0, np.pi, mesh)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    for i in range(mesh):
        for j in range(mesh):
            point = np.array([x[i][j],y[i][j],z[i][j]])
            all_min_dist = distance.cdist([point], points, "euclidean")
            min_dist = np.min(all_min_dist)
            Z[i][j] = min_dist
            #print(i,j,Z[i][j])

    ax.contourf(X, Y, Z, 50,alpha = 0.8, cmap=plt.get_cmap('jet'))
