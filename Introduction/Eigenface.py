import cv2 as cv
import os
import numpy as np
from numpy import linalg as lin
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors



# get the matrix
def loadImageSet(path):
    # create a matrix to contain all image vectors
    FaceMat = np.zeros((135, 195*231))

    # my own face recognition
    # FaceMat = np.zeros((144, 195 * 231))

    j =0
    # get all the image from the directory and reshape them into the matrix
    for i in os.listdir(path):
        img = cv.imread(path + i, 0)
        FaceMat[j, :] = img.reshape(1, -1)[0, :]
        j += 1
    return FaceMat.T


# get the mean face
def mean_face(path):
    # get the face matrix
    matrix = loadImageSet(path)
    # get the mean vector
    mean = np.mean(matrix, axis=1)
    # reshape the mean vector
    new = np.reshape(mean, (231, 195))
    # convert the matrix into ndarray
    mean_face = np.array(new, dtype=np.uint8)
    return mean_face


# PCA algorithm
def pca(path, k=10):
    # get all face into a matrix
    matrix1 = loadImageSet(path)
    dimension, m = matrix1.shape[0], matrix1.shape[1]
    mean = np.mean(matrix1, axis=1)

    # normalise the matrix
    matrix = np.zeros((dimension, m), dtype=np.float)
    for i in range(m):
        matrix[:, i] = matrix1[:, i] - mean

    # compute the covariance matrix
    # covMatrix = np.dot(matrix, matrix.T) / dimension
    # the faster way
    covMatrix = np.dot(matrix.T, matrix) / m

    # compute the eigenvalues and eigenvectors
    eigenvalue, eigenvector = np.linalg.eig(covMatrix)

    # sort the eigenvalues
    index = np.argsort(-eigenvalue)

    # keep the top k eigenvectors
    eigenvector = np.dot(matrix, eigenvector[:, index[:k]])
    return matrix, eigenvector


# face recognition
def face_recognition(testfaces, matrix, eigenvectors, meanFace):
    m = matrix.shape[1]
    k = eigenvectors.shape[1]
    # create index to contain all the similar results
    index = np.zeros((10, 3), dtype=np.int)

    # contain all the selected features
    features = np.zeros((m, k))
    for j in range(m):
        for i in range(k):
            features[j, i] = np.dot(eigenvectors[:, i].T, matrix[:, j])

    # normalized the test faces
    for i in range(k):
        testfaces[:, i] = testfaces[:, i] - meanFace

    # contains all the test features
    test_features = np.zeros((10, k))
    for j in range(10):
        for i in range(k):
            test_features[j, i] = np.dot(eigenvectors[:, i].T, testfaces[:, j])

    # determine the test faces' projection on the features space and find the three most similar face
    for j in range(10):
        test_features = np.dot(testfaces[:, j].T, eigenvectors)
        # contains the square of distance between each test feature and all features in feature space
        distance = []
        for i in range(m):
            distance.append(np.sum(np.power((test_features - features[i, :]), 2)))
        pick = np.argsort(distance)
        index[j, :] = pick[:3]
    return index


# get the mean face and display it
meanface = mean_face('Yale-FaceA\\trainingset\\')
# cv.imshow('', meanface)
# cv.waitKey()

# flat the mean face
meanface = meanface.flatten()
diffMatrix, vectors = pca('Yale-FaceA\\trainingset\\')

# ghost face
for i in range(10):
    vector = vectors[:, i]

    # map the vector into range of 0 and 255
    vector = 255 * (vector - np.min(vector))/(np.max(vector) - np.min(vector))

    vector = np.reshape(vector, (231, 195))
    vector = np.array(vector, dtype=np.uint8)
    plt.subplot(2, 5, i+1), plt.imshow(vector, cmap='gray'), plt.title('Eigenface')
    plt.xticks([]), plt.yticks([])
plt.show()

# obtain all the test image
testfaces = []
for i in os.listdir('Yale-FaceA\\testset\\'):
    testfaces.append(i)

# get all the faces
faces = loadImageSet('Yale-FaceA\\trainingset\\')

# face recognition
test = loadImageSet('Yale-FaceA\\testset\\')
ind = face_recognition(test, diffMatrix, vectors, meanface)

# the current number of test face
count = 0

# display the similar faces
for name in testfaces:
    testface = cv.imread('Yale-FaceA\\testset\\' + name, 0)
    ind1, ind2, ind3 = ind[count, 0], ind[count, 1], ind[count, 2]
    first, second, third = faces[:, ind1], faces[:, ind2], faces[:, ind3]
    firstface = np.array(np.reshape(first, (231, 195)), dtype=np.uint8)
    secondface = np.array(np.reshape(second, (231, 195)), dtype=np.uint8)
    thirdface = np.array(np.reshape(third, (231, 195)), dtype=np.uint8)
    count = count + 1
    plt.subplot(1, 4, 1), plt.imshow(testface, cmap='gray'), plt.title('test face')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 2), plt.imshow(firstface, cmap='gray'), plt.title('similar face one')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 3), plt.imshow(secondface, cmap='gray'), plt.title('similar face two')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 4), plt.imshow(thirdface, cmap='gray'), plt.title('similar face three')
    plt.xticks([]), plt.yticks([])
    plt.show()
