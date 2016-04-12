# -*- coding: utf-8 -*-
"""
W16 EECS 445 - Introduction to Machine Learning
Project 2- "BOT-ANY"
Skeleton Code 
"""

import numpy as np
import project2 as p2
from scipy import stats
import matplotlib.pyplot as plt


class Point(object):
    """
    Represents a data point
    """
    def __init__(self, name, label, originalAttrs):
        """
        Initialize name, label and attributes
        """
        self.name = name
        self.label = label
        self.attrs = originalAttrs

    def dimensionality(self):
        """Returns dimension of the point"""
        return len(self.attrs)
    
    def getAttrs(self):
        """Returns attr"""
        return self.attrs
    
    def distance(self, other):
        """
        other: point, to which we are measuring distance to 
        Return Euclidean distance of this point with other
        """
        #Euclidean distance metric
        return np.linalg.norm(self.attrs-other.getAttrs())

    def getName(self):
        """Returns name"""
        return self.name
    
    def getLabel(self):
        """Returns label"""
        return self.label


class Cluster(object):
    """
    A Cluster is defined as a set of Points
    """
    def __init__(self, points):
        """
        Points of a cluster are saved in a list, self.points
        """
        self.points = points
        
    def getPoints(self):
        """Returns points in the cluster as a list"""
        return self.points
    
    def getPurity(self):
        """Returns number of unique labels and most common label in cluster"""
        labels=[]
        for p in self.points:
            labels.append(p.getLabel())
        
        cluster_label,count=stats.mode(labels)
        return len(labels), np.float64(count)

    def equivalent(self,other):
        """
        other: Cluster, what we are comparing this Cluster to
        Returns true if both Clusters are equivalent, or false otherwise
        """
        if len(self.getPoints())!=len(other.getPoints()):
            return False

        matched=[] 
        for p1 in self.getPoints():
            for point2 in other.getPoints():
                if p1.distance(point2)==0 and point2 not in matched:
                    matched.append(point2)
        if len(matched) == len(self.getPoints()):
            return True
        else:
            return False

    def removePoint(self,point):
        """Remove given point from cluster"""
        self.points.remove(point)
                           
    def getCentroid(self):
        """ TODO """
        """Returns the centroid of the cluster"""
        attrs = [p.getAttrs() for p in self.getPoints()]
        return Point('Centroid',None,np.mean(attrs,axis=0))

    def getMedoid(self):
        """ TODO """
        """
        Returns the point in the cluster that is the closest to all other points
        in the cluster.
        """
        medoid = None
        smallest_dist = float('inf')

        for p1 in self.points:
            dist = 0.0
            for p2 in self.points:
                dist += p1.distance(p2)
            if dist < smallest_dist:
                smallest_dist = dist
                medoid = p1

        assert medoid != None  # sanity check
        return medoid
        
    def singleLinkageDist(self, other):
        """ TODO """
        """
        other: Cluster, what we are comparing this Cluster to
        
        Returns the float distance between the points that 
        are closest to each other, where one point is from
        self and the other point is from other. Uses the 
        Euclidean dist between 2 points, defined in Point.
        """
        minDist = float("inf")
        for p1 in self.points:
            for p2 in other.points:
                dist = p1.distance(p2)
                if dist < minDist:
                    minDist = dist

        return minDist

    def maxLinkageDist(self, other):
        """ TODO """
        """
        other: Cluster, what we are comparing this Cluster to
        
        Returns the float distance between the points that 
        are farthest from each other, where one point is from 
        self and the other point is from other. Uses the 
        Euclidean dist between 2 points, defined in Point.
        """
        maxDist = 0.0
        for p1 in self.points:
            for p2 in other.points:
                dist = p1.distance(p2)
                if dist > maxDist:
                    maxDist = dist

        return maxDist

    def averageLinkageDist(self, other):
        """ TODO """
        """
        other: Cluster, what we are comparing this Cluster to
        
        Returns the float average (mean) distance between all 
        pairs of points, where one point is from self and the 
        other point is from other. Uses the Euclidean dist 
        between 2 points, defined in Point.
        """
        all_dists = []
        for p1 in self.points:
            for p2 in other.points:
                all_dists.append(p1.distance(p2))

        return np.mean(all_dists)
      

class ClusterSet(object):
    """
    A ClusterSet is defined as a list of clusters
    """
    def __init__(self):
        """
        Initialize an empty set, without any clusters
        """
        self.members = []

    def add(self, c):
        """
        c: Cluster
        Appends a cluster c to the end of the cluster list
        only if it doesn't already exist in the ClusterSet.
        If it is already in self.members, raise a ValueError
        """
        if c in self.members:
            raise ValueError
        self.members.append(c)
        
    def getClusters(self):
        """Returns clusters in the ClusterSet"""
        return self.members[:]
    
    def numClusters(self):
        """Returns number of clusters in the ClusterSet"""
        return len(self.members)
            
    def getScore(self):
        """
            Returns accuracy of the clustering given by the clusters
            in theClusterSet object
        """
        total_correct=0
        total=0
        for c in self.members:
            n,n_correct=c.getPurity()
            total=total+n
            total_correct=total_correct+n_correct

        return total_correct/float(total)     

    def equivalent(self,other):
        """ 
        other: another ClusterSet object

        Returns true if both ClusterSets are equivalent, or false otherwise
        """
        if len(self.getClusters())!=len(other.getClusters()):
            return False       

        matched=[]
        for c1 in self.getClusters():
            for c2 in other.getClusters():
                if c1.equivalent(c2) and c2 not in matched:
                    matched.append(c2)
        if len(matched) == len(self.getClusters()):
            return True
        else:
            return False

    def getCentroids(self):
        """ TODO """
        """Returns centroids of each cluster in the ClusterSet as a list"""

        return [c.getCentroid() for c in self.getClusters()]
        
    def getMedoids(self):
        """ TODO """
        """Returns medoids of each cluster in the ClusterSet as a list"""

        return [c.getMedoid() for c in self.members]
        
    def mergeClusters(self, c1, c2):
        """ TODO """
        """
        c1: Cluster, in self.members
        c2: Cluster, in self.members

        Adds a new Cluster containing the union of c1 and c2
        to self.members.
        Removes c1 and c2 from self.members
        """
        c_union = Cluster(list(set(c1.getPoints()) | set(c2.getPoints())))
        self.add(c_union)
        self.members.remove(c1)
        self.members.remove(c2)
    
    def findClosest(self, linkage):
        """ TODO """
        """
        linkage: method, linkage criteria to be used
        
        Returns a tuple containing the two most similar clusters
        in self.members, which will be the two closest clusters
        with the smallest value as determined by the linkage criteria
        """
        closest_dist = float('inf')
        toMerge = None,None

        num_clusters = self.numClusters()
        for i in range(num_clusters):
            for j in range(i+1,num_clusters):
                c1 = self.members[i]
                c2 = self.members[j]
                dist = linkage(c1,c2)

                if dist < closest_dist:
                    closest_dist = dist
                    toMerge = c1,c2

        assert toMerge != None,None  # sanity check
        return toMerge
    
    def mergeOne(self, linkage):
        """ TODO """
        """
        linkage: method, linkage criteria to be used

        Merges the two most similar clusters in self.members

        Returns a tuple containing the clusters that were merged
        """
        toMerge = self.findClosest(linkage)
        self.mergeClusters(toMerge[0],toMerge[1])
        
        return toMerge
    

def buildFlowerImagePoints(X,y):
    """
    X : (n,d) feature matrix, in which each row represents an image
    y: (n,1) array, vector, containing labels corresponding to X 
    Returns a list of Points 
    """
    (n,d)=X.shape
    images={}
    points = []
    for i in xrange(0,n):
        if y[i] not in images.keys():
            images[y[i]]=[]
        images[y[i]].append(X[i,:])
    for flower in images.keys():
        count=0
        for im in images[flower]:
            points.append(Point(str(flower)+'_'+str(count),flower,im))
            count=count+1

    return points    


def plotClusters(clusters,title,average):
    """
    clusters: a ClusterSet object
    title: title of the plot
    average: A method the ClusterSet class which determines 
            how to calculate average of the points in the cluster
            Can take values ClusterSet.getCentroids or ClusterSet.getMedoids
            
    Plots the clusters given by ClusterSet object along with the average points
    of each cluster
    """
    plt.figure()
    np.random.seed(20)
    label=0
    colors={}
    centroids=average(clusters)
    for c in centroids:
        coord=c.getAttrs()
        plt.plot(coord[0],coord[1],'ok',markersize=12)
    for c in clusters.getClusters():   
        label=label+1
        colors[label]=np.random.rand(3,)
        for p in c.getPoints():
            coord=p.getAttrs()
            plt.plot(coord[0],coord[1],'o',c=colors[label])
    plt.title(title)
    plt.show()


def generate2dPoints(N, source=None):
    """
    Generate 3 clusters each with N points
    Returns a list of point objects
    
    source takes two values 'file' or None
    By default N points are generated randomly
    otherwise if source=='file', 20 pts for
    each cluster are loaded from a file
    
    """
    mu=[[0,0.5],[1,1],[2,0.5]]
    sigma=[[0.1,0.1],[0.25,0.25],[0.15,0.15]]
    label=0
    points=[]
    count = 0
    if source == 'file':
        arr = np.genfromtxt('toydataset.txt')
    for m,s in zip(mu,sigma):
        label=label+1
        for i in range(N):
            if source == 'file':
                x = arr[count,:]
                count+=1

            elif source == None:
                x=p2.randomSample2d(m,s)
            
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


def randomInit(points,k):
    """ TODO """
    """ 
        points: a list of point objects
        k: Number of initial centroids/medoids
        Returns a list of k unique points randomly selected from points
    """
    return np.random.choice(points,k,replace=False)


def cheatInit(points):
    """ TODO """
    """
    points: a list of point objects
    Returns a list containing the medoids of each cluster
    determined by labels of the points
    """
    unique_labels = np.unique([p.getLabel() for p in points])
    cluster_set = ClusterSet()
    for label in unique_labels:
        cluster_set.add(Cluster([p for p in points if p.label == label]))
    
    return cluster_set.getMedoids()

def bonusInit(points,k):
    """ TODO """
    # This is the initialization algorithm used by k-means++
    centroids = np.random.choice(points,1,replace=False).tolist()
    k -= 1
    while k > 0:
        point_dists = {}
        # for each point in set of non-centroid points
        total_dist_sq = 0.0
        for p in [point for point in points if point not in centroids]:
            dist_squared = np.amin([p.distance(c) for c in centroids])**2
            point_dists[p] = dist_squared
            total_dist_sq += dist_squared
        # update all dists to reflect weights
        relevant_pts = []
        weights = []
        for key,value in point_dists.items():
            relevant_pts.append(key)
            weights.append(value/total_dist_sq)

        next = np.random.choice(relevant_pts,1,replace=False,p=weights)[0]
        centroids.append(next)
        k -= 1
    return centroids
     
    
def kMeans(points,k,init='random',plot='off',average=ClusterSet.getCentroids):
    # Note: the specification lists 'average' as a function parameter, but it
    # wasn't included in the function definition. I added it in order to make
    # kMedoids() easier
    """ TODO """
    """
    points: a list of Point objects
    k: the number of clusters we want to end up with
    init: The method of initialization, takes two valus 'cheat'
          and 'random'. If init='cheat', then use cheatInit to get
          initial clusters. If init='random', then use randomInit
          to initialize clusters. Default value 'random'.
    plot: Takes two values, 'on' and 'off' , If plot='on' plots
          clusters along with corresponding centroids for each 
          iteration of the algorithm.
    average: A method of the ClusterSet class which determines 
            how to calculate average of the points in the cluster
            Can take values ClusterSet.getCentroids or ClusterSet.getMedoids
            Default value ClusterSet.getCentroids
    
    Clusters points into k clusters using kMeans clustering.

    Returns an instance of ClusterSet corresponding to k clusters
    """
    if init == 'random':
        centroids = randomInit(points,k)
    elif init == 'cheat':
        centroids = cheatInit(points)
    else:
        centroids = bonusInit(points,k)
    kMeansClusters = ClusterSet()

    equivalent_set = False

    num_iter = 0
    while not equivalent_set:
        points_lists = [[] for i in range(k)]  # k lists of points for k Cluster()s
        num_iter += 1
        # Assign clusters
        for point in points:
            # distances from each centroid
            distances = [point.distance(centroid) for centroid in centroids]
            idx = np.argmin(distances)
            points_lists[idx].append(point)
        newClusters = ClusterSet()
        for l in points_lists: newClusters.add(Cluster(l))

        # get new Centroids
        centroids = average(newClusters)

        # update condition
        if kMeansClusters.equivalent(newClusters):
            equivalent_set = True
        else:
            kMeansClusters = newClusters

        if plot == 'on':
            kType = 'kMeans' if average==ClusterSet.getCentroids \
                             else 'kMedoids'
            title = '%s, %s, iter: %d' % (kType, init, num_iter)
            plotClusters(kMeansClusters,title,average)

    return kMeansClusters
    
    
def kMedoids(points,k,init='random',plot='off'):
    """ TODO """
    """
    points: a list of Point objects
    k: the number of clusters we want to end up with
    init: The method of initialization, takes two valus 'cheat'
          and 'random'. If init='cheat', then use cheatInit to get 
          initial clusters. If init='random', then use randomInit
          to initialize clusters. Default value 'random'.
    plot: Takes two values, 'on' and 'off' , If plot='on' plots
          clusters along with corresponding centroids for each 
          iteration of the algorithm.
    
    Clusters points into k clusters using kMedoids clustering.

    Returns an instance of ClusterSet corresponding to k clusters
    """
    return kMeans(points,k,init,plot,ClusterSet.getMedoids)

    
def applyPCAfromEig(X,U,l,mu):
    """ Note: PCA assumes the data are centered, i.e. we need to subtract the mean"""
    """
    X: n by d array representing n d-dimensional data points
    U: d by d array representing d d-dimensional eigenvectors;
       each column is a unit eigenvector; sorted by eigenvalue
    l: number of principal components
    mu: d by 1 array representing mean of X 
   
    Returns: Z, Ul
             Z: n by l array X represented by the first l principal components
             Ul: d by l array with the first l eigenvectors
    """
    X = np.subtract(X,mu) # subtract mean
    Ul = np.take(U,range(l),axis=1) # take the first l columns for U
    Z = np.dot(X,Ul)
    return Z, Ul


def reconstructFromPCA(Z,U,mu):
    """ Note: We used PCA on centered data, don't forget to add the mean back"""
    """
    Z: n by l array representing n l-dimensional data points
    U: d by l array representing l d-dimensional eigenvectors;
       each column is a unit eigenvector; sorted by eigenvalue
    mu: d by 1 array representing mean of X 
   
    Returns: X_rec n by d array representing reconstructed images
    """
    X_rec = np.dot(Z,U.T)
    X_rec = np.add(X_rec,mu)
    return X_rec
    

def hCluster(points, linkage, numClusters, plot='off'):
    """ TODO """
    """
    points: a list of Point objects
    linkage: the metric used to determine which Clusters to merge,
             for example, Cluster.singleLinkageDist. We can then
             use linkage(cluster1, cluster2) within this function.
    numClusters: the number of clusters we want to end up with
    plot: Takes two values, 'on' and 'off' , If plot='on' plots
          clusters along with corresponding centroids for the 
          final iteration
    
    Computes the clusters generated by hierarchical clustring

    Returns hClusters an instance of ClusterSet corresponding to a clustering of points
    
    """
    assert numClusters > 0  # sanity check

    hClusters = ClusterSet()
    # add Clusters of 1 point into hClusters
    for p in points:
        hClusters.add(Cluster([p]))

    while hClusters.numClusters() > numClusters:
        hClusters.mergeOne(linkage)

    # Fpr plot title
    if linkage == Cluster.singleLinkageDist:
        link = 'Single'
    elif linkage == Cluster.maxLinkageDist:
        link = 'Max'
    else:
        link = 'Average'

    if plot == 'on':
        plotClusters(hClusters,'Hierarchical, %s' % (link), \
                     ClusterSet.getCentroids)

    return hClusters


if __name__ == "__main__":
    X, y = p2.get_data(['data/Giant', 'data/Jerusalem', 'data/Swamp'])
    E,mu = p2.PCA(X)

    plots = 'on'
    points = generate2dPoints(20,source='file')
    flower_points = buildFlowerImagePoints(X,y)
    k = 3

    """ 1 """
    p2.showIm(X[0])
    p2.showIm(X[1])
    """
    p2.showIm(mu)
    p2.plotGallery([p2.vecToImage(E[:,i]) for i in range(12)])
    l_range = [1,10,50,100,500,1600]
    
    for l in l_range:
        print l  # for reference
        Z,Ul = applyPCAfromEig(X,E,l,mu)
        X_rec = reconstructFromPCA(Z,Ul,mu)
        p2.plotGallery([p2.vecToImage(X_rec[i,:]) for i in range(12)])
    """

    """ 2c """
    """
    clusters_single = hCluster(points,Cluster.singleLinkageDist,k,plot=plots)
    clusters_max = hCluster(points,Cluster.maxLinkageDist,k,plot=plots)
    clusters_average = hCluster(points,Cluster.averageLinkageDist,k,plot=plots)
    """

    """ 3d """
    """
    kMeans(points,k,init='random',plot=plots)
    kMeans(points,k,init='cheat',plot=plots)
    kMedoids(points,k,init='random',plot=plots)
    kMedoids(points,k,init='cheat',plot=plots)
    """

    """ 4a """
    """
    kMeans_scores = []
    cluster_set = None
    for i in range(10):
        cluster_set = kMeans(flower_points,k,init='random',plot='off')
        kMeans_scores.append(cluster_set.getScore())
    print 'Average: ', np.average(kMeans_scores)
    print 'Minimum: ', np.amin(kMeans_scores)
    print 'Maximum: ', np.amax(kMeans_scores)
    # plot medoids
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())

    kMedoids_scores = []
    for i in range(10):
        cluster_set = kMedoids(flower_points,k,init='random',plot='off')
        kMedoids_scores.append(cluster_set.getScore())
    print 'Average: ', np.average(kMedoids_scores)
    print 'Minimum: ', np.amin(kMedoids_scores)
    print 'Maximum: ', np.amax(kMedoids_scores)
    # plot medoids
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())
    """

    """ 4b """
    """
    print 'Single'
    cluster_set = hCluster(flower_points,Cluster.singleLinkageDist,k,plot='off')
    print 'Performance: ', cluster_set.getScore()
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())

    print 'Max'
    cluster_set = hCluster(flower_points,Cluster.maxLinkageDist,k,plot='off')
    print 'Performance: ', cluster_set.getScore()
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())

    print 'Average'
    cluster_set = hCluster(flower_points,Cluster.averageLinkageDist,k,plot='off')
    print 'Performance: ', cluster_set.getScore()
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())
    """

    """ 4c """
    """
    scores_kMeans = []
    scores_kMedoids = []
    scores_hierarch = []

    l_range = range(1,80,5)
    for l in l_range:
        Z,Ul = applyPCAfromEig(X,E,l,mu)
        X_rec = reconstructFromPCA(Z,Ul,mu)
        flower_points = buildFlowerImagePoints(X_rec,y)
        cluster_set = kMeans(flower_points,k,init='cheat',plot='off')
        scores_kMeans.append(cluster_set.getScore())
        cluster_set = kMedoids(flower_points,k,init='cheat',plot='off')
        scores_kMedoids.append(cluster_set.getScore())
        cluster_set = hCluster(flower_points,Cluster.maxLinkageDist,k,plot='off')
        scores_hierarch.append(cluster_set.getScore())

    # create plot of scores for all three algorithms
    line_kMeans, = plt.plot(l_range,scores_kMeans,marker='o')
    line_kMedoids, = plt.plot(l_range,scores_kMedoids,marker='o')
    line_hierarch, = plt.plot(l_range,scores_hierarch,marker='o')
    plt.title('Score vs. Number of Principal Components')
    plt.xlabel('$l$')
    plt.ylabel('Score')
    plt.legend([line_kMeans,line_kMedoids,line_hierarch], \
               ['K-means','K-medoids','Hierarchical'], loc='upper left')
    plt.show()
    """

    """ 4d """
    """
    dirs = ['data/Giant', 'data/Jerusalem', 'data/Swamp', 'data/Daisy']
    flower_counts = [30,30,40,12]
    splits = np.add.accumulate(flower_counts)
    splits = np.delete(splits,len(splits)-1)  # don't split past the end
    X_all,y_all = p2.get_data(dirs,np.sum(flower_counts))
    [X_g,X_j,X_s,X_d] = np.split(X_all,splits)
    [y_g,y_j,y_s,y_d] = np.split(y_all,splits)

    points_dg = buildFlowerImagePoints(np.vstack((X_d,X_g)), \
                                       np.hstack((y_d,y_g)))
    points_dj = buildFlowerImagePoints(np.vstack((X_d,X_j)), \
                                       np.hstack((y_d,y_j)))
    points_ds = buildFlowerImagePoints(np.vstack((X_d,X_s)), \
                                       np.hstack((y_d,y_s)))
    k = 2
    scores_dg = []
    scores_dj = []
    scores_ds = []
    for i in range(20):
        clusters_dg = kMeans(points_dg,k,init='random',plot='off')
        scores_dg.append(clusters_dg.getScore())
        clusters_dj = kMeans(points_dj,k,init='random',plot='off')
        scores_dj.append(clusters_dj.getScore())
        clusters_ds = kMeans(points_ds,k,init='random',plot='off')
        scores_ds.append(clusters_ds.getScore())
    print "Daisy vs. Giant: ", np.average(scores_dg)
    print "Daisy vs. Jerusalem: ", np.average(scores_dj)
    print "Daisy vs. Swamp: ", np.average(scores_ds)
    """

    """ Bonus """
    """
    kMeans_scores = []
    cluster_set = None
    for i in range(10):
        cluster_set = kMeans(flower_points,k,init='bonus',plot='off')
        kMeans_scores.append(cluster_set.getScore())
    print 'Average: ', np.average(kMeans_scores)
    print 'Minimum: ', np.amin(kMeans_scores)
    print 'Maximum: ', np.amax(kMeans_scores)
    # plot medoids
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())

    kMedoids_scores = []
    for i in range(10):
        cluster_set = kMedoids(flower_points,k,init='bonus',plot='off')
        kMedoids_scores.append(cluster_set.getScore())
    print 'Average: ', np.average(kMedoids_scores)
    print 'Minimum: ', np.amin(kMedoids_scores)
    print 'Maximum: ', np.amax(kMedoids_scores)
    # plot medoids
    for medoid in cluster_set.getMedoids():
        p2.showIm(medoid.getAttrs())
    """