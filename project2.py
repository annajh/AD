from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

# Input:
# dir_names: a list of directory names. From each directory a set of 
# pictures belonging to the same cluster is processed.
# no_images: Number of total images in all directories mentioned in dir_names 
#
# Returns:
# X: an n x d array, in which each row represents an image
# y: a n X 1 vector, elements of which are integers between 0 and nc-1
#    where nc is the number of classes represented in the data

def get_data(dir_names, no_images=100):
    X = np.zeros((no_images,1600))
    y = np.zeros(no_images)
    count = 0
    for label, dict_name in enumerate(dir_names):
        for filename in glob.glob(dict_name+'/*.png'):
            # convert('LA') includes alpha channel
            img = Image.open(filename).convert('L')
            X[count,:] = np.asarray(img).flatten()
            y[count] = label
            count+=1
    return X, y

# Input
# im: a row or column vector of dimension d
# size: a pair of positive integers (i, j) such that i * j = d
#       defaults to the right value for our images
# Opens a new window and displays the image
imageSize = (40,40)
def showIm(im, size = imageSize):
    plt.figure()
    im = im.copy()
    im.resize(*size)
    plt.imshow(im.astype(float), cmap = cm.gray)
    plt.show()

# Take an eigenvector and make it into an image
def vecToImage(x, size = imageSize):
  im = x/np.linalg.norm(x)
  im = im*(256./np.max(im))
  im.resize(*size)
  return im

# Plot an array of images
# Input
# - images: a 12 by d array
# - title: string title for whole window
# - subtitles: a list of 12 strings to be used as subtitles for the
#              subimages, or an empty list
# - h, w, n_row, n_col: can be used for other image sizes or other
#           numbers of images in the gallery

def plotGallery(images, title='plot', subtitles = [],
                 h=40, w=40, n_row=3, n_col=4):
    plt.figure(title,figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min(len(images), n_row * n_col)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.xticks(())
        plt.yticks(())    
    plt.show()

# Perform PCA
# Input:
# - X: n by d array representing n d-dimensional data points
# Output:
# - u: d by d array representing d d-dimensional eigenvectors of the empirical
#       covariance matrix of X;
#      each column is a unit eigenvector; sorted by eigenvalue
# - mu: 1 by d array representing the mean of the input data
# This version uses SVD for better numerical performance when d >> n

def PCA(X, sphere = False):
    (n, d) = X.shape
    mu = np.mean(X, axis=0)
    (x, l, v) = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    u = np.array([vi/(li if (sphere and li>1.0e-10) else 1.0) \
                  for (li, vi) \
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return u, mu


# Randomly samples a point from a normal distribution having mean mu
# and standard deviation sigma
# Returns the sampled point
def randomSample2d(mu,sigma):
    x=np.random.normal(mu[0],sigma[0])
    y=np.random.normal(mu[1],sigma[1])
    return np.array([x,y])

# Get the first image for each label
def getRepImage(X, y, label):
    tmp = X[y == label, :]
    return vecToImage(tmp[0, :])




