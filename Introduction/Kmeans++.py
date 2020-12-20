import cv2 as cv
import numpy as np
from numpy import *
# from sklearn.cluster import KMeans
import random
from matplotlib import pyplot as plt
import scipy


# k-means algorithm
def my_kmeans(points, clusters, centers=None):
    rows, cols = points.shape
    # create random centers of clusters
    centers = np.zeros((clusters, cols), np.float)
    for i in range(clusters):
        randPos = random.randint(0, rows)
        currentPos = points[randPos]
        centers[i] = currentPos
    # create a matrix whose first column contains the index of points and the second column contains the
    pointIndex = np.zeros((rows, 2), np.float)
    # perform k-means
    while True:
        # save the original centers to compare current centers later
        originalCenters = centers
        for i in range(rows):
            minDist = inf
            minIndex = -1
            # find the nearest data point
            for j in range(clusters):
                currentDist = sum(power(centers[j, :] - points[i, :], 2))
                # if the distance of current point and center is less than previous points, minDist and minIndex will be changed
                if currentDist < minDist:
                    minDist = currentDist
                    minIndex = j
            # save this data point with its index and distance to the nearest center
            pointIndex[i, :] = minIndex, minDist
        # update the centers
        for cent in range(clusters):
            # data points of the same cluster
            ptsInClust = points[nonzero(pointIndex[:, 0] == cent)[0]]
            # the average center of each cluster
            centers[cent, :] = mean(ptsInClust, axis=0)
        # whether the center has changed
        if (originalCenters == centers).all():
            break
    # index of points
    pointIndex = pointIndex[:, 0]
    return pointIndex


# get the nearest distance between two vectors
def nearest(point, cluster_centers):
    min_dist = np.Inf
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        # compute the distance between current data and each center
        d = sum(np.power(point - cluster_centers[i], 2))
    # keep the nearest distance
    if min_dist > d:
        min_dist = d
    return min_dist


# use kmeans++ to get all the centers
def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k , n)))
    # pick one center randomly
    index = np.random.randint(0, m)
    cluster_centers[0] = np.copy(points[index])
    # use this list to save the distance
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # find the nearest center to current data
            d[j] = nearest(points[j], cluster_centers[0:i])
            # add all the distance up
            sum_all += d[j]
        sum_all *= np.random.random()
        # set the farthest data as new center
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j])
            break
    return cluster_centers


# create the 5-D vector
def createVector(src):
    height, width = src.shape[0], src.shape[1]
    size = height * width
    l, a, b = cv.split(cv.cvtColor(src, cv.COLOR_RGB2LAB))
    l = l
    a, b = a, b
    vector = []
    for i in range(height):
        for j in range(width):
            v = []
            v.append(l[i, j])
            v.append(a[i, j])
            v.append(b[i, j])
            v.append(20 * float(i)/height)
            v.append(20 * float(j)/width)
            vector.append(v)
    return np.mat(vector)


# create the r, g, b vector
def rgbVector(src):
    height, width = src.shape[0], src.shape[1]
    size = height * width
    b, g, r = cv.split(src)
    vector = []
    for i in range(height):
        for j in range(width):
            v = []
            v.append(r[i, j])
            v.append(g[i, j])
            v.append(b[i, j])
            vector.append(v)
    return np.mat(vector)


# # show the clustering result
# def showClusters(data):
#     row, col = data.shape
#     pic_new = np.ones((row, col), dtype=np.uint8)
#     # compute the value of each pixel
#     for i in range(row):
#         for j in range(col):
#             pic_new[i, j] = int(256 / (data[i][j] + 0.1))
#     return pic_new


img = plt.imread('peppers.png', 1)
row, col = img.shape[0], img.shape[1]
v = createVector(img)
c = get_centroids(v, 5)
clusters = my_kmeans(v, 5, c)
clusters = clusters.reshape([row, col])
# new = showClusters(clusters)
plt.imshow(clusters)
plt.show()
# cv.imshow('', new)
# cv.waitKey()


